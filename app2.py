# --- Streamlit App for Conversational Q&A with Parent Document Retriever (PDR) using LangChain and Gemini LLM ---

import streamlit as st
import os
import time
import hashlib
import logging

# LangChain components
# --- Chain for conversational RAG ---
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
# --- Parent Document Retriever components ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore # Simple in-memory store for parent docs
from langchain.vectorstores.base import VectorStore # Import base class for type checking

# Google Generative AI SDK
import google.generativeai as genai

# --- Configuration Constants ---
# --- Rationale: Centralizing configuration makes the app easier to manage and tune. ---
# Directories
PDF_DIR = "uploaded_pdfs"           # Directory to save uploaded PDF files
VECTORSTORE_BASE_DIR = "vector_stores" # Base directory to store Chroma vector databases

# Embedding Model (Sentence Transformers are efficient for local embeddings)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Good balance of performance/size

# --- PDR Specific Chunking ---
# Define size for larger chunks (parents) - adjust as needed
PARENT_CHUNK_SIZE = 2000 # Experiment with this for sections like References
PARENT_CHUNK_OVERLAP = 200
# Define size for smaller chunks (children) - used for embedding/searching
CHILD_CHUNK_SIZE = 400  # Smaller chunks for more precise embedding matching
CHILD_CHUNK_OVERLAP = 50

# Gemini LLM Settings
# Using Flash as per user's confirmation, but Pro is also an option if available
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.7  # Controls creativity (0=deterministic, >1=more creative)

# --- Setup Logging ---
# --- Rationale: Logging helps track application flow and debug issues without cluttering the UI. ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Create directories ---
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
    # --- Rationale: Ensures each distinct PDF has its own isolated vector store for child chunks. ---
    return os.path.join(VECTORSTORE_BASE_DIR, pdf_hash + "_child_chunks")

# --- Core Logic Functions ---

def load_vectorstore_if_exists(vs_path, embedding_function):
    """Attempts to load an existing Chroma vector store."""
    if os.path.exists(vs_path):
        try:
            logger.info(f"Attempting to load existing Chroma vector store from {vs_path}")
            vectorstore = Chroma(persist_directory=vs_path, embedding_function=embedding_function)
            vectorstore.similarity_search("test", k=1) # Quick check
            logger.info(f"Successfully loaded Chroma vector store from {vs_path}")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load vector store from {vs_path}: {e}. Will need to recreate.", exc_info=True)
    return None # Return None if path doesn't exist or loading fails

# --- Rationale: Custom prompt for ConversationalRetrievalChain to guide condensation ---
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the follow up question is already a standalone question or unrelated to the conversation history, return it as is.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# --- Rationale: Updated system prompt for main QA, includes formatting instructions ---
QA_SYSTEM_PROMPT_TEMPLATE = """You are a helpful and detailed Q&A assistant. Use the following pieces of retrieved context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the context does not contain the answer, state clearly: "The provided context does not contain the answer to this question."
Keep the answer concise but comprehensive, based *only* on the provided context.
Do NOT use any external knowledge or information beyond the provided context.

**Formatting Instructions:**
- When asked for detailed explanations, summaries, or lists, structure your answer clearly.
- Use markdown formatting like headings (`## Heading`), bullet points (`* Point`), and bold text (`**bold**`) where appropriate to improve readability.
- If extracting a list (like references), present it as an ordered or unordered list based on the context.

CONTEXT:
{context}

User Question: {question}
Assistant Answer (based ONLY on context):"""

def init_session_state():
    """Initializes all necessary variables in Streamlit's session state."""
    # LLM Initialization (using Gemini)
    if 'llm' not in st.session_state:
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY") # Or st.secrets.get("GOOGLE_API_KEY")
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

    # Memory for conversational context
    if 'memory' not in st.session_state:
        # Return messages=True for ConversationalRetrievalChain
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key='answer'
        )

    # Retriever (will hold the ParentDocumentRetriever)
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    # Document Store for PDR (Parent chunks)
    if 'docstore' not in st.session_state:
        # Using InMemoryStore for simplicity. For large docs or persistence across
        # restarts without reprocessing, consider LocalFileStore or RedisStore.
        st.session_state.docstore = InMemoryStore()

    # QA Chain (will hold ConversationalRetrievalChain)
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # Other state variables
    if 'chat_history_display' not in st.session_state: st.session_state.chat_history_display = [] # For display purposes
    if 'processed_pdf_path' not in st.session_state: st.session_state.processed_pdf_path = None
    if 'processed_pdf_hash' not in st.session_state: st.session_state.processed_pdf_hash = None
    if 'current_doc_name' not in st.session_state: st.session_state.current_doc_name = "No document loaded"


# --- Main Streamlit App ---
init_session_state()

# --- Apply Custom Font (Overpass) ---
# --- Rationale: Uses CSS injection via Markdown for cross-browser font application. ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Overpass:wght@400;700&display=swap');
html, body, [class*="css"] {
   font-family: 'Overpass', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Document Q&A with PDR & Conversation")
st.caption(f"Using: LangChain + Gemini + ParentDocRetriever | Doc: {st.session_state.current_doc_name}")

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
        st.session_state.retriever = None # Reset retriever specifically
        st.session_state.qa_chain = None
        st.session_state.chat_history_display = []
        if 'memory' in st.session_state: st.session_state.memory.clear()
        # Re-initialize docstore for the new file
        st.session_state.docstore = InMemoryStore()
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
            st.session_state.processed_pdf_path = None
            st.session_state.processed_pdf_hash = None
            st.session_state.current_doc_name = "File saving error"
            needs_processing = False
            st.stop()

    # --- Process if needed (Setup PDR) ---
    if needs_processing:
        with st.status(f"Processing {uploaded_file.name} (setting up Parent Document Retriever)..."):
            try:
                st.write(f"📄 Reading PDF: {uploaded_file.name}")
                loader = PyPDFLoader(st.session_state.processed_pdf_path)
                docs = loader.load()
                if not docs: raise ValueError("PyPDFLoader failed to load data.")

                # --- Initialize PDR Components ---
                parent_store = st.session_state.docstore # Use the one from session state
                embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                vectorstore_path = get_vectorstore_path(st.session_state.processed_pdf_hash)

                st.write(f"🔨 Initializing Retriever components...")
                logger.info("Initializing PDR components...")

                # Try loading existing vectorstore first (contains child embeddings)
                vectorstore = load_vectorstore_if_exists(vectorstore_path, embedding_function)

                # --- FIX: Ensure vectorstore is initialized if loading fails/doesn't exist ---
                vectorstore_needs_adding = False
                if vectorstore is None:
                    logger.info(f"No existing vector store found or load failed. Initializing new Chroma instance for PDR at {vectorstore_path}")
                    # Initialize an empty Chroma instance pointing to the correct path
                    vectorstore = Chroma(
                        persist_directory=vectorstore_path,
                        embedding_function=embedding_function
                    )
                    vectorstore_needs_adding = True # Flag that we need to add docs

                # Check if vectorstore is indeed a VectorStore instance now
                if not isinstance(vectorstore, VectorStore):
                     raise TypeError(f"Vectorstore initialization failed. Expected VectorStore, got {type(vectorstore)}")


                # Define splitters
                parent_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
                )
                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
                )

                # Initialize ParentDocumentRetriever (NOW vectorstore is guaranteed to be a VectorStore instance)
                logger.debug("Initializing ParentDocumentRetriever object...")
                pdr = ParentDocumentRetriever(
                    vectorstore=vectorstore, # Pass the loaded or newly initialized store
                    docstore=parent_store,  # Pass the store for parent docs
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )

                # Add documents ONLY if we created a new vectorstore instance OR if docstore is empty
                # This populates the vectorstore with child embeddings and the docstore with parent chunks
                should_add_docs = vectorstore_needs_adding or (not list(parent_store.yield_keys()))

                if should_add_docs:
                    st.write(f"✨ Indexing document chunks (this might take a while)...")
                    logger.info("Adding documents to ParentDocumentRetriever...")
                    pdr.add_documents(docs, ids=None) # Let PDR handle splitting, embedding, storing

                    # Persist vectorstore changes explicitly AFTER adding documents if it was newly created
                    if vectorstore_needs_adding and hasattr(vectorstore, 'persist'):
                        logger.info(f"Persisting vector store changes to {vectorstore_path}...")
                        vectorstore.persist()
                        logger.info("Vector store persisted.")
                else:
                    logger.info("Vector store loaded and docstore seems populated. Skipping add_documents.")


                st.session_state.retriever = pdr # Store the initialized PDR
                st.session_state.current_doc_name = uploaded_file.name
                logger.info(f"Parent Document Retriever ready for {uploaded_file.name}")
                st.write("✅ Document processing complete.")

            except Exception as e:
                logger.error(f"Error during PDR setup or processing: {e}", exc_info=True)
                st.error(f"An error occurred during document processing/PDR setup: {e}")
                # Reset state
                st.session_state.retriever = None
                st.session_state.qa_chain = None
                st.session_state.processed_pdf_path = None
                st.session_state.processed_pdf_hash = None
                st.session_state.current_doc_name = "Processing error"
                st.stop()
        st.rerun() # Rerun after successful processing

    else: # File hash matches, not new processing needed initially
        # --- Logic to handle app restart with existing file ---
        # Ensure retriever is initialized if it's None but should be ready
        if not st.session_state.retriever and st.session_state.processed_pdf_path and st.session_state.llm:
             logger.info("Retriever not found in session state, attempting re-initialization for existing file...")
             try:
                 parent_store = st.session_state.docstore # Reuse existing docstore from state
                 embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                 vectorstore_path = get_vectorstore_path(st.session_state.processed_pdf_hash)

                 # Attempt to load the vector store, it *should* exist
                 vectorstore = load_vectorstore_if_exists(vectorstore_path, embedding_function)

                 if vectorstore and isinstance(vectorstore, VectorStore):
                     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
                     child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)

                     pdr = ParentDocumentRetriever(
                         vectorstore=vectorstore,
                         docstore=parent_store, # Use existing docstore from session state
                         child_splitter=child_splitter,
                         parent_splitter=parent_splitter,
                     )

                     # Check and populate docstore if empty (important for InMemoryStore on restart)
                     if not list(parent_store.yield_keys()):
                         logger.info("Populating docstore on re-initialization...")
                         loader_temp = PyPDFLoader(st.session_state.processed_pdf_path)
                         docs_temp = loader_temp.load()
                         if docs_temp:
                              # Use add_documents, but only add to docstore if vectorstore already exists
                              pdr.add_documents(docs_temp, ids=None, add_to_docstore=True)
                              logger.info("Docstore repopulated.")
                         else:
                              logger.error("Failed to reload docs to populate empty docstore on restart.")

                     st.session_state.retriever = pdr
                     logger.info("Successfully re-initialized Parent Document Retriever.")
                     st.rerun() # Rerun after successful re-init
                 else:
                     # If vector store load fails even on restart, something is wrong
                     logger.error(f"Cannot re-initialize retriever: Failed to load vector store at {vectorstore_path}")
                     st.warning("Could not re-initialize Q&A system. Vector store missing or corrupt. Please re-upload the PDF.")
                     # Reset hash to force reprocessing on next interaction
                     st.session_state.processed_pdf_hash = None


             except Exception as e:
                 logger.error(f"Error re-initializing PDR: {e}", exc_info=True)
                 st.warning("Error during Q&A system re-initialization. Try re-uploading.")

        # Update doc name just in case state was lost
        if st.session_state.current_doc_name != uploaded_file.name and st.session_state.processed_pdf_hash == uploaded_hash:
             st.session_state.current_doc_name = uploaded_file.name
        logger.debug(f"File {uploaded_file.name} (hash: {uploaded_hash}) already processed.")


# --- Setup QA Chain (ConversationalRetrievalChain with PDR) ---
# --- Rationale: Handles chat history and rephrases questions for better retrieval. ---
if st.session_state.retriever and st.session_state.llm and not st.session_state.qa_chain:
    logger.info("Attempting to set up ConversationalRetrievalChain...")
    try:
        # Define the prompt for the QA part of the chain
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(QA_SYSTEM_PROMPT_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=st.session_state.retriever, # Use the ParentDocumentRetriever
            memory=st.session_state.memory,       # Use the conversational memory
            return_source_documents=True,         # Return PDR's parent chunks
            condense_question_prompt=CONDENSE_QUESTION_PROMPT, # Rephrase question prompt
            combine_docs_chain_kwargs={"prompt": qa_prompt}, # Main QA prompt
            # verbose=True # Optional: for debugging chain steps
        )
        logger.info("ConversationalRetrievalChain successfully created.")
        st.rerun() # Ensure UI updates

    except Exception as e:
        logger.error(f"Failed to create ConversationalRetrievalChain: {e}", exc_info=True)
        st.error(f"Error setting up the Q&A system: {e}. Check logs.")
        st.session_state.qa_chain = None

# Display warning if retriever setup failed or is missing
elif uploaded_file and not st.session_state.retriever and st.session_state.llm:
     st.warning("⚠️ Document possibly processed, but retriever setup failed or is missing. Check logs.")


# --- Display Chat History ---
# Uses the separate display history list
for msg_data in st.session_state.chat_history_display: # Corrected loop variable
    with st.chat_message(msg_data["role"]):
        st.markdown(msg_data["content"])
        # --- FIX: Check current message data for sources ---
        if msg_data["role"] == "assistant" and "sources" in msg_data and msg_data["sources"]:
             with st.expander("View Sources Used"):
                 # Iterate through the sources stored IN THE CURRENT msg_data dictionary
                 for i, doc in enumerate(msg_data["sources"]): # Corrected access
                     page_num = doc.metadata.get('page', 'N/A')
                     # PDR adds parent doc metadata ('doc_id') to child chunks if using default ID scheme
                     parent_id = doc.metadata.get('doc_id', 'N/A')
                     source_info = f"Source {i+1} (Page: {page_num}, Parent ID: {parent_id})"
                     preview_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                     st.info(f"{source_info}:\n```\n{preview_content}\n```")


# --- Q&A Interaction Logic ---
if st.session_state.qa_chain: # Only show input if the chain is ready
    if user_input := st.chat_input(f"Ask about '{st.session_state.current_doc_name}'..."):

        # Add user message to display history
        st.session_state.chat_history_display.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            source_documents = []
            start_time = time.time()

            # Use thinking indicator
            with st.spinner("🤔 Thinking & Retrieving..."):
                try:
                    logger.info(f"Invoking ConversationalRetrievalChain for query: '{user_input}'")
                    # Call the conversational chain (expects 'question' key from memory setup)
                    response = st.session_state.qa_chain({"question": user_input})
                    end_time = time.time()

                    result_text = response.get('answer', "Error: Could not retrieve answer.")
                    source_documents = response.get('source_documents', [])

                    logger.info(f"ConversationalRetrievalChain execution time: {end_time - start_time:.2f} seconds")
                    logger.info(f"Retrieved {len(source_documents)} source documents via PDR.")

                except Exception as e:
                    end_time = time.time()
                    logger.error(f"Error during QA chain execution ({end_time - start_time:.2f}s): {e}", exc_info=True)
                    result_text = f"⚠️ Sorry, an error occurred while processing your question: {e}"
                    message_placeholder.error(result_text)

            # Display final response
            if 'response' in locals(): # Check response exists (no exception)
                 message_placeholder.markdown(result_text)

            # Store assistant response for display history
            # Note: 'memory' object already stores history for the chain itself
            assistant_message_display = {"role": "assistant", "content": result_text, "sources": source_documents}
            st.session_state.chat_history_display.append(assistant_message_display)

            # Display sources immediately
            if source_documents:
                 with st.expander("View Sources Used"):
                      # Use the sources from the *current* response dict
                     for i, doc in enumerate(assistant_message_display["sources"]):
                         page_num = doc.metadata.get('page', 'N/A')
                         parent_id = doc.metadata.get('doc_id', 'N/A') # PDR specific metadata
                         source_info = f"Source {i+1} (Page: {page_num}, Parent ID: {parent_id})"
                         preview_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                         st.info(f"{source_info}:\n```\n{preview_content}\n```")

# --- Handling Initial/Error States ---
elif not uploaded_file:
    st.info("✨ Upload a PDF document to begin asking questions.")
    # Reset state if needed
    if st.session_state.processed_pdf_hash:
        logger.info("No file uploaded currently, resetting previous state.")
        for key in ['retriever', 'qa_chain', 'processed_pdf_path', 'processed_pdf_hash', 'current_doc_name', 'chat_history_display', 'memory', 'docstore']:
            if key in st.session_state:
                if key == 'chat_history_display': st.session_state[key] = []
                elif key == 'memory' and hasattr(st.session_state[key], 'clear'): st.session_state[key].clear()
                elif key == 'docstore' and hasattr(st.session_state[key], 'clear'): st.session_state[key].clear() # Clear InMemoryStore if needed
                elif key == 'current_doc_name': st.session_state[key] = "No document loaded"
                else: st.session_state[key] = None
        st.rerun()

elif st.session_state.processed_pdf_path and not st.session_state.qa_chain:
     st.warning("⚠️ Document processed, but Q&A system is not ready. Check logs (API Key/LLM/Retriever setup).")