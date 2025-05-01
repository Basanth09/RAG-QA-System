
import streamlit as st
import os
import time
import hashlib
import logging

# LangChain components
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# Import retrievers for Hybrid Search
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.schema import Document # Only needed if manually creating Documents for BM25

# Google Generative AI SDK
import google.generativeai as genai

# --- Configuration Constants ---
# --- Rationale: Centralizing configuration makes the app easier to manage and tune. ---
# Directories
PDF_DIR = "uploaded_pdfs"           # Directory to save uploaded PDF files
VECTORSTORE_BASE_DIR = "vector_stores" # Base directory to store Chroma vector databases

# Embedding Model (Sentence Transformers are efficient for local embeddings)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Good balance of performance/size

# Text Splitting (Controls how PDFs are chunked for embedding)
CHUNK_SIZE = 500           # Size of text chunks (experiment based on document structure)
CHUNK_OVERLAP = 100        # Overlap between chunks to maintain context

# Retriever Settings (Controls how many chunks are retrieved by each part of the ensemble)
FINAL_RETRIEVER_K = 3      # How many results each retriever (BM25, Vector) should return

# Gemini LLM Settings
# Using Flash as per user's confirmation, but Pro is also an option if available
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or "gemini-1.5-pro-latest", "gemini-2.5-pro-preview-03-25"
GEMINI_TEMPERATURE = 0.7  # Controls creativity (0=deterministic, >1=more creative)

# --- Setup Logging ---
# --- Rationale: Logging helps track application flow and debug issues without cluttering the UI. ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Create directories if they don't exist ---
# --- Rationale: Ensures the application has the necessary folders to store files and data. ---
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)

# --- Helper Functions ---
# --- Rationale: Encapsulating reusable logic improves code readability and maintainability. ---

def compute_file_hash(file_bytes):
    """Computes the SHA256 hash of file bytes to uniquely identify file content."""
    return hashlib.sha256(file_bytes).hexdigest()

def get_vectorstore_path(pdf_hash):
    """Generates a unique path for the vector store based on the PDF content hash."""
    # --- Rationale: Ensures each distinct PDF has its own isolated vector store. ---
    return os.path.join(VECTORSTORE_BASE_DIR, pdf_hash)

# --- Core Logic Functions ---

def load_or_create_chroma_vectorstore(pdf_path, pdf_hash, splits_for_creation=None):
    """
    Loads a Chroma vector store if one exists for the PDF hash, otherwise creates it.
    Now primarily handles the Chroma DB part, assuming splits are provided for creation.

    Args:
        pdf_path (str): The path to the PDF file (used for logging).
        pdf_hash (str): The SHA256 hash of the PDF file content.
        splits_for_creation (list[Document], optional): Pre-split documents needed ONLY if creating store.

    Returns:
        Chroma: The loaded or newly created Chroma vector store object, or None on failure.
    """
    vectorstore_path = get_vectorstore_path(pdf_hash)
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(vectorstore_path):
        try:
            logger.info(f"Attempting to load existing Chroma vector store from {vectorstore_path}")
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embedding_function
            )
            vectorstore.similarity_search("test query", k=1) # Quick check
            logger.info(f"Successfully loaded Chroma vector store for hash {pdf_hash}.")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load vector store from {vectorstore_path}: {e}. Will attempt recreation if splits provided.", exc_info=True)
            # Fall through to creation logic IF splits_for_creation is provided

    # If store doesn't exist or loading failed, create a new one *if splits are provided*
    if splits_for_creation:
        logger.info(f"Creating new Chroma vector store for {os.path.basename(pdf_path)} (hash: {pdf_hash}) at {vectorstore_path}")
        try:
            vectorstore = Chroma.from_documents(
                documents=splits_for_creation,
                embedding=embedding_function,
                persist_directory=vectorstore_path
            )
            logger.info(f"Successfully created and persisted Chroma vector store at {vectorstore_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error during Chroma vector store creation for {pdf_path}: {e}", exc_info=True)
            st.error(f"An error occurred during vector store creation: {e}")
            return None
    else:
        # Cannot create if store didn't exist and no splits were provided
        logger.error(f"Vector store not found at {vectorstore_path} and no splits provided for creation.")
        return None


def init_session_state():
    """Initializes all necessary variables in Streamlit's session state."""
    if 'template' not in st.session_state:
        # Refined prompt allowing grounded synthesis
        st.session_state.template = """SYSTEM: You are a factual Q&A assistant. Your sole purpose is to accurately answer questions based *ONLY* on the provided context. Do NOT use any prior knowledge or external information.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

INSTRUCTIONS:
1. Analyze the USER QUESTION.
2. Carefully examine the provided CONTEXT to find all relevant information.
3. Construct the answer by synthesizing information found *only* within the provided CONTEXT. If the answer requires combining information from multiple parts of the context, ensure the connection is logically supported by the text and the combined information accurately addresses the user's question.
4. State *only* what the text supports, even if it requires connecting related sentences. Do not add external knowledge or make assumptions beyond the text.
5. The answer must be complete and include *all* relevant details found in the CONTEXT that address the USER QUESTION.
6. If specific terms, names, numbers, or examples are mentioned in the question, verify them against the CONTEXT and include them in the answer *if* they are present and relevant.
7. If the CONTEXT contains information about a process or mechanism related to the question, describe it fully as detailed in the CONTEXT.
8. If the CONTEXT does *not* contain the answer to the USER QUESTION, state *exactly*: "The document does not provide this information." Do not attempt to guess or infer.
9. Do NOT add any introductory phrases, concluding remarks, or conversational filler. Focus solely on presenting the factual answer derived from the CONTEXT.
10. Do NOT mention the CONTEXT itself in the answer (e.g., don't say "According to the context..."). Just provide the answer.
11. Be precise and avoid ambiguity. If the context implies a relationship (e.g., cause and effect), state it clearly based *only* on the textual evidence provided.

ASSISTANT ANSWER:"""

    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template
        )

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history", return_messages=True, input_key="question"
        )

    if 'llm' not in st.session_state:
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            # google_api_key = st.secrets.get("GOOGLE_API_KEY") # Alternative

            if not google_api_key:
                st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable or Streamlit secrets.")
                logger.error("GOOGLE_API_KEY not found in environment or secrets.")
                st.session_state.llm = None
            else:
                genai.configure(api_key=google_api_key)
                st.session_state.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME,
                    temperature=GEMINI_TEMPERATURE,
                    convert_system_message_to_human=True
                )
                logger.info(f"LLM initialized: {GEMINI_MODEL_NAME}.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM ({GEMINI_MODEL_NAME}): {e}", exc_info=True)
            st.error(f"Failed to initialize LLM ({GEMINI_MODEL_NAME}). Check API Key/access. Error: {e}")
            st.session_state.llm = None

    # Initialize other state variables
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state: st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state: st.session_state.qa_chain = None
    if 'processed_pdf_path' not in st.session_state: st.session_state.processed_pdf_path = None
    if 'processed_pdf_hash' not in st.session_state: st.session_state.processed_pdf_hash = None
    if 'current_doc_name' not in st.session_state: st.session_state.current_doc_name = "No document loaded"
    # Add state for storing document splits needed for BM25
    if 'current_document_splits' not in st.session_state: st.session_state.current_document_splits = None

# --- Main Streamlit App ---
init_session_state()

st.title("🧠 Enhanced Document Q&A with RAG implemented Hybrid Search")
st.caption(f"Using: LangChain + Gemini + Chroma + BM25 | Doc: {st.session_state.current_doc_name}")

# --- File Uploader and Processing ---
uploaded_file = st.file_uploader(
    "Upload a PDF document:", type='pdf', key="pdf_uploader"
)

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    uploaded_hash = compute_file_hash(bytes_data)
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    needs_processing = False

    # --- Check if file is new or changed ---
    if st.session_state.processed_pdf_hash != uploaded_hash:
        logger.info(f"New file detected: {uploaded_file.name} (Hash: {uploaded_hash}).")
        needs_processing = True

        # Reset state for the new document
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.session_state.current_document_splits = None # Clear previous splits
        if 'memory' in st.session_state: st.session_state.memory.clear()
        st.session_state.current_doc_name = "Processing..."

        # Save the new file
        try:
            with open(file_path, "wb") as f: f.write(bytes_data)
            logger.info(f"Saved uploaded file to {file_path}")
            st.session_state.processed_pdf_path = file_path
            st.session_state.processed_pdf_hash = uploaded_hash
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}", exc_info=True)
            st.error(f"Failed to save uploaded file: {e}")
            # Reset relevant states on save failure
            st.session_state.processed_pdf_path = None
            st.session_state.processed_pdf_hash = None
            st.session_state.current_doc_name = "File saving error"
            needs_processing = False
            st.stop()

    # --- Process if needed (Load, Split, Create/Load Vector Store) ---
    if needs_processing:
        with st.status(f"Processing {uploaded_file.name}..."):
            try:
                # 1. Load and Split Document
                st.write(f"📄 Reading & splitting PDF...")
                loader = PyPDFLoader(st.session_state.processed_pdf_path)
                data = loader.load()
                if not data: raise ValueError("PyPDFLoader failed to load data.")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                all_splits = text_splitter.split_documents(data)
                if not all_splits: raise ValueError("Text splitting resulted in no documents.")
                st.session_state.current_document_splits = all_splits # Store splits in session state
                logger.info(f"Split PDF into {len(all_splits)} chunks.")
                st.write(f"Split into {len(all_splits)} document chunks.") # Corrected English

                # 2. Create/Load Chroma Vector Store
                st.write("🔢 Indexing document for vector search (Chroma)...")
                # Pass splits only needed if vector store needs creation
                vs = load_or_create_chroma_vectorstore(
                    st.session_state.processed_pdf_path,
                    st.session_state.processed_pdf_hash,
                    splits_for_creation=all_splits # Pass splits here for potential creation
                )
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.current_doc_name = uploaded_file.name
                    logger.info(f"Chroma vector store ready for {uploaded_file.name}")
                    st.write("✅ Document processing complete.")
                else:
                    raise ValueError("Failed to create or load Chroma vector store.")

            except Exception as e:
                logger.error(f"Error during processing steps: {e}", exc_info=True)
                st.error(f"An error occurred during document processing: {e}")
                # Reset state fully on processing error
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
                st.session_state.processed_pdf_path = None
                st.session_state.processed_pdf_hash = None
                st.session_state.current_document_splits = None
                st.session_state.current_doc_name = "Processing error"
                st.stop()
        st.rerun() # Rerun after successful processing

    else: # File hash matches, not new
        # Ensure splits are available if app restarted with existing file
        if not st.session_state.current_document_splits and st.session_state.processed_pdf_path:
            try:
                logger.info("Reloading/Splitting document for BM25 (app restart)...")
                loader = PyPDFLoader(st.session_state.processed_pdf_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                all_splits = text_splitter.split_documents(data)
                if all_splits:
                    st.session_state.current_document_splits = all_splits
                    logger.info(f"Successfully reloaded {len(all_splits)} splits.")
                else:
                     logger.warning("Failed to get splits on reload.")
            except Exception as e:
                logger.error(f"Could not reload/resplit document on restart: {e}")
                st.warning("Could not prepare data for hybrid search after app restart.")
        # Update doc name just in case
        if st.session_state.current_doc_name != uploaded_file.name:
             st.session_state.current_doc_name = uploaded_file.name
        logger.debug(f"File {uploaded_file.name} (hash: {uploaded_hash}) already processed.")


# --- Setup QA Chain (Hybrid Search - Ensemble Retriever) ---
if st.session_state.vectorstore and \
   st.session_state.llm and \
   st.session_state.current_document_splits and \
   not st.session_state.qa_chain: # Ensure all components are ready

    logger.info("Attempting to set up the QA chain with EnsembleRetriever (Hybrid Search)...")
    try:
        # 1. Initialize BM25 Retriever (Keyword Search)
        logger.debug("Initializing BM25 Retriever...")
        bm25_retriever = BM25Retriever.from_documents(
            st.session_state.current_document_splits # Use the splits from session state
        )
        bm25_retriever.k = FINAL_RETRIEVER_K # Retrieve top K based on keywords

        # 2. Initialize Vector Store Retriever (Semantic Search)
        logger.debug("Initializing Chroma Vector Retriever...")
        chroma_vector_retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={'k': FINAL_RETRIEVER_K} # Retrieve top K based on semantics
        )

        # 3. Initialize Ensemble Retriever
        logger.debug("Initializing Ensemble Retriever...")
        # --- Rationale: Combines keyword and semantic results using Reciprocal Rank Fusion by default. ---
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_vector_retriever],
            weights=[0.5, 0.5] # Adjust weights: 0.5/0.5 gives equal importance
        )

        # 4. Create the RetrievalQA Chain
        logger.debug("Initializing RetrievalQA chain...")
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=ensemble_retriever, # Use the ensemble retriever!
            verbose=False,
            chain_type_kwargs={
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory
            },
            return_source_documents=True
        )
        logger.info("QA chain with EnsembleRetriever successfully created.")
        st.rerun() # Rerun to update UI state (e.g., enable chat input)

    except Exception as e:
        logger.error(f"Failed to create QA chain with EnsembleRetriever: {e}", exc_info=True)
        st.error(f"Error setting up Q&A system with Hybrid Search: {e}. Check logs.")
        st.session_state.qa_chain = None # Ensure chain is None on failure

# Add warning if splits are missing, preventing chain setup
elif st.session_state.vectorstore and st.session_state.llm and not st.session_state.current_document_splits and not st.session_state.qa_chain:
     logger.warning("Cannot setup QA chain: Document splits not available for BM25.")
     st.warning("Document processed, but data for hybrid search setup is missing. Try re-uploading the file.")


# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
             # Corrected English label
             with st.expander("View Sources Used"):
                 for i, doc in enumerate(message["sources"]):
                     page_num = doc.metadata.get('page', 'N/A')
                     # Corrected English label
                     source_info = f"Source {i+1} (Page: {page_num})"
                     # Display limited content to avoid clutter
                     preview_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                     st.info(f"{source_info}:\n```\n{preview_content}\n```")


# --- Q&A Interaction Logic ---
if st.session_state.qa_chain: # Only show input if the chain is ready
    if user_input := st.chat_input(f"Ask about '{st.session_state.current_doc_name}'..."):

        # Add user message
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            source_documents = []
            start_time = time.time()

            # Corrected English spinner text
            with st.spinner("🧠 Thinking..."):
                try:
                    logger.info(f"Invoking QA chain (Ensemble) for query: '{user_input}'")
                    response = st.session_state.qa_chain({"query": user_input})
                    end_time = time.time()

                    result_text = response.get('result', "Error: Could not retrieve answer.")
                    source_documents = response.get('source_documents', [])

                    logger.info(f"QA chain execution time: {end_time - start_time:.2f} seconds")
                    logger.info(f"Retrieved {len(source_documents)} source documents via EnsembleRetriever.")

                except Exception as e:
                    end_time = time.time()
                    logger.error(f"Error during QA chain execution ({end_time - start_time:.2f}s): {e}", exc_info=True)
                    # Corrected English error message
                    result_text = f"⚠️ Sorry, an error occurred while processing your question: {e}"
                    message_placeholder.error(result_text)

            # Display final response
            if 'response' in locals():
                 message_placeholder.markdown(result_text)

            # Store history and display sources
            assistant_message = {"role": "assistant", "message": result_text, "sources": source_documents}
            st.session_state.chat_history.append(assistant_message)

            if source_documents:
                 # Corrected English expander label
                 with st.expander("View Sources Used"):
                     for i, doc in enumerate(source_documents):
                         page_num = doc.metadata.get('page', 'N/A')
                         # Corrected English source label
                         source_info = f"Source {i+1} (Page: {page_num})"
                         preview_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                         st.info(f"{source_info}:\n```\n{preview_content}\n```")

# --- Handling Initial/Error States ---
elif not uploaded_file:
    # Corrected English info message
    st.info("✨ Upload a PDF document to begin asking questions.")
    # Reset state if needed
    if st.session_state.processed_pdf_hash:
        logger.info("No file uploaded currently, resetting previous state.")
        for key in ['vectorstore', 'qa_chain', 'processed_pdf_path', 'processed_pdf_hash', 'current_doc_name', 'chat_history', 'memory', 'current_document_splits']:
            if key in st.session_state:
                if key == 'chat_history': st.session_state[key] = []
                elif key == 'memory' and hasattr(st.session_state[key], 'clear'): st.session_state[key].clear()
                elif key == 'current_doc_name': st.session_state[key] = "No document loaded"
                else: st.session_state[key] = None
        st.rerun()

elif st.session_state.processed_pdf_path and not st.session_state.qa_chain:
     # Corrected English warning message
     st.warning("⚠️ Document processed, but Q&A system is not ready. Check logs (API Key/LLM/Splits availability).")