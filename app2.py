# --- Streamlit App for Conversational Q&A with Parent Document Retriever (PDR) using LangChain and Gemini LLM ---
# Complete Code (app.py) - Tuned PDR Chunking, Stricter Prompt, InMemoryStore

import streamlit as st
import os
import time
import hashlib
import logging

# LangChain components
# --- Chain for conversational RAG ---
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
# --- Updated Imports based on Deprecation Warnings ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
# --- Core LangChain components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
# --- Parent Document Retriever components ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore # Using InMemoryStore
from langchain.vectorstores.base import VectorStore # Import base class for type checking

# Google Generative AI SDK
import google.generativeai as genai

# --- Configuration Constants ---
# Directories
PDF_DIR = "uploaded_pdfs"
VECTORSTORE_BASE_DIR = "vector_stores_pdr" # Base dir for Chroma (child chunks)

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# --- PDR Specific Chunking ---
PARENT_CHUNK_SIZE = 750 # Keeping parent chunk size large for context
PARENT_CHUNK_OVERLAP = 100
# --- CHANGE: Smaller child chunks for potentially better precision ---
CHILD_CHUNK_SIZE = 200
CHILD_CHUNK_OVERLAP = 30

# Gemini LLM Settings
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.5 # Keep slightly lower temp for potentially more factual output

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Create directories ---
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)

# --- Helper Functions ---
def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def get_vectorstore_path(pdf_hash):
    """Path for Chroma DB (child embeddings)."""
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
    return None

# --- Prompt for Condensing Question ---
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the follow up question is already a standalone question or unrelated to the conversation history, return it as is.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# --- *** FURTHER REFINED SYSTEM PROMPT FOR REASONING & ADHERENCE *** ---
QA_SYSTEM_PROMPT_TEMPLATE = """You are an expert Q&A assistant designed for deep analysis and accurate information synthesis.
Your goal is to answer the user's question based *exclusively* on the provided context documents.
Do NOT use any prior knowledge or external information whatsoever. Your knowledge is strictly limited to the CONTEXT provided.

**Reasoning Process:**
1.  **Understand the Question:** Fully grasp all parts and nuances of the user's question.
2.  **Identify Relevant Context:** Carefully scan the provided CONTEXT documents. Identify *all* sentences, paragraphs, or data points that *directly* relate to the question. Pay close attention to details, numbers, names, and relationships explicitly mentioned.
3.  **Synthesize Information:** If multiple pieces of context are relevant, synthesize them into a coherent answer. Do not just list isolated facts. Explain connections or relationships *only if they are explicitly supported by the text in the CONTEXT*.
4.  **Address Complexity:** If the question is complex, break it down and address each part systematically based *only* on the information found in the CONTEXT.
5.  **Handle Lack of Information:** If the CONTEXT does not contain the necessary information to answer the question *completely and accurately*, explicitly state: "Based on the provided context, I cannot answer that question." or "The provided context does not contain specific details about [the specific topic asked]". Do not speculate, infer beyond the text, or make assumptions.
6.  **Format for Clarity:** Structure your answer clearly using markdown (headings (`## Heading`), bullet points (`* Point`), bold text (`**bold**`)) when appropriate, especially for detailed explanations or lists (like references).
7.  **Extract Lists:** If extracting a list (like references or bibliography), present it as an ordered or unordered list based on the context. Ensure *all* relevant items found *within the provided CONTEXT* are listed.

**CRITICAL Constraints:**
- Base answers *ONLY* and *STRICTLY* on the provided CONTEXT.
- You MUST NOT mention information external to the provided CONTEXT. Do NOT make up details or use outside knowledge.
- If the CONTEXT explicitly contains the answer (e.g., a specific name, title, number, date), you MUST use the exact information from the CONTEXT in your answer. Do NOT paraphrase explicitly stated facts like titles or names.
- Be precise and objective. If the context is unclear or insufficient, state that.

CONTEXT:
{context}

User Question: {question}
Assistant Answer (based ONLY on context):"""

def init_session_state():
    """Initializes all necessary variables in Streamlit's session state."""
    if 'llm' not in st.session_state:
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                st.error("Google API Key not found...") ; logger.error("GOOGLE_API_KEY not found...")
                st.session_state.llm = None
            else:
                genai.configure(api_key=google_api_key)
                st.session_state.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME, temperature=GEMINI_TEMPERATURE, convert_system_message_to_human=True
                )
                logger.info(f"LLM initialized: {GEMINI_MODEL_NAME}.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM ({GEMINI_MODEL_NAME}): {e}", exc_info=True)
            st.error(f"Failed to initialize LLM ({GEMINI_MODEL_NAME}). Check API Key/access. Error: {e}")
            st.session_state.llm = None

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key='answer'
        )

    if 'retriever' not in st.session_state: st.session_state.retriever = None
    if 'docstore' not in st.session_state: st.session_state.docstore = InMemoryStore()
    if 'qa_chain' not in st.session_state: st.session_state.qa_chain = None
    if 'chat_history_display' not in st.session_state: st.session_state.chat_history_display = []
    if 'processed_pdf_path' not in st.session_state: st.session_state.processed_pdf_path = None
    if 'processed_pdf_hash' not in st.session_state: st.session_state.processed_pdf_hash = None
    if 'current_doc_name' not in st.session_state: st.session_state.current_doc_name = "No document loaded"

# --- Main Streamlit App ---
init_session_state()

# Apply Custom Font
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Overpass:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Overpass', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Advanced Document Q&A (PDR + Reasoning)")
st.caption(f"Using: LangChain + Gemini + PDR | Doc: {st.session_state.current_doc_name}")

uploaded_file = st.file_uploader("Upload PDF:", type='pdf', key="pdf_uploader")

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    uploaded_hash = compute_file_hash(bytes_data)
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    needs_processing = False

    # --- Check if file is new or changed ---
    if st.session_state.processed_pdf_hash != uploaded_hash:
        logger.info(f"New file detected: {uploaded_file.name} (Hash: {uploaded_hash}).")
        needs_processing = True
        # Reset state
        st.session_state.retriever = None; st.session_state.qa_chain = None
        st.session_state.chat_history_display = []; st.session_state.memory.clear()
        st.session_state.docstore = InMemoryStore(); st.session_state.current_doc_name = "Processing..."
        # Save file
        try:
            with open(file_path, "wb") as f: f.write(bytes_data)
            logger.info(f"Saved uploaded file to {file_path}")
            st.session_state.processed_pdf_path = file_path
            st.session_state.processed_pdf_hash = uploaded_hash
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}", exc_info=True); st.error(f"Failed to save file: {e}")
            st.session_state.processed_pdf_path = None; st.session_state.processed_pdf_hash = None
            st.session_state.current_doc_name = "File saving error"; needs_processing = False; st.stop()

    # --- Process if needed (Setup PDR with InMemoryStore) ---
    if needs_processing:
        with st.status(f"Processing {uploaded_file.name} (Parent Document Retriever)..."):
            try:
                st.write(f"📄 Reading PDF...")
                loader = PyMuPDFLoader(st.session_state.processed_pdf_path)
                docs = loader.load()
                if not docs: raise ValueError("PyMuPDFLoader failed to load data.")

                # --- Initialize PDR Components ---
                parent_store = st.session_state.docstore
                embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                vectorstore_path = get_vectorstore_path(st.session_state.processed_pdf_hash)

                st.write(f"🔨 Initializing Retriever components...")
                logger.info("Initializing PDR components...")

                vectorstore = load_vectorstore_if_exists(vectorstore_path, embedding_function)

                vectorstore_needs_adding = False
                if vectorstore is None:
                    logger.info(f"No existing vector store found. Initializing new Chroma instance at {vectorstore_path}")
                    vectorstore = Chroma(
                        persist_directory=vectorstore_path, embedding_function=embedding_function
                    )
                    vectorstore_needs_adding = True
                    if not isinstance(vectorstore, VectorStore):
                        raise TypeError(f"Vectorstore init failed. Got {type(vectorstore)}")

                # --- Use Updated Chunk Sizes ---
                parent_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
                )
                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP # Using 250/40 now
                )

                logger.debug("Initializing ParentDocumentRetriever object...")
                pdr = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=parent_store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )

                # Add documents if vectorstore is new OR docstore is empty
                should_add_docs = vectorstore_needs_adding or (not list(parent_store.yield_keys()))
                if should_add_docs:
                    st.write(f"✨ Indexing document chunks (can take time)...")
                    logger.info("Adding documents via ParentDocumentRetriever...")
                    pdr.add_documents(docs, ids=None)
                    logger.info("Finished adding documents to PDR.")

                    if vectorstore_needs_adding and hasattr(vectorstore, 'persist'):
                        logger.info(f"Persisting vector store changes to {vectorstore_path}...")
                        vectorstore.persist()
                        logger.info("Vector store persisted.")
                else:
                    logger.info("Vector store loaded and docstore seems populated. Skipping add_documents.")

                st.session_state.retriever = pdr
                st.session_state.current_doc_name = uploaded_file.name
                logger.info(f"Parent Document Retriever ready for {uploaded_file.name}")
                st.write("✅ Document processing complete.")

            except Exception as e:
                logger.error(f"Error during PDR setup/processing: {e}", exc_info=True)
                st.error(f"Error during document processing/PDR setup: {e}")
                st.session_state.retriever = None; st.session_state.qa_chain = None
                st.session_state.processed_pdf_path = None; st.session_state.processed_pdf_hash = None
                st.session_state.current_doc_name = "Processing error"
                if 'docstore' in st.session_state: st.session_state.docstore.clear()
                st.stop()
        st.rerun()

    else: # File hash matches
        # --- Logic to handle app restart with existing file ---
        if not st.session_state.retriever and st.session_state.processed_pdf_path and st.session_state.llm:
             logger.info("Retriever not found in session state, attempting re-initialization...")
             try:
                 if 'docstore' not in st.session_state: st.session_state.docstore = InMemoryStore()
                 parent_store = st.session_state.docstore
                 embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                 vectorstore_path = get_vectorstore_path(st.session_state.processed_pdf_hash)
                 vectorstore = load_vectorstore_if_exists(vectorstore_path, embedding_function)

                 if vectorstore and isinstance(vectorstore, VectorStore):
                     # --- Use Updated Chunk Sizes ---
                     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
                     child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)

                     pdr = ParentDocumentRetriever(
                         vectorstore=vectorstore, docstore=parent_store,
                         child_splitter=child_splitter, parent_splitter=parent_splitter,
                     )

                     if not list(parent_store.yield_keys()):
                         logger.info("Populating InMemory docstore on re-initialization...")
                         loader_temp = PyMuPDFLoader(st.session_state.processed_pdf_path)
                         docs_temp = loader_temp.load()
                         if docs_temp:
                              pdr.add_documents(docs_temp, ids=None, add_to_docstore=True)
                              logger.info("Docstore repopulated.")
                         else: logger.error("Failed to reload docs to populate empty docstore.")

                     st.session_state.retriever = pdr
                     logger.info("Successfully re-initialized PDR.")
                     st.rerun()
                 else:
                     logger.error(f"Cannot re-initialize: Failed to load vector store at {vectorstore_path}.")
                     st.warning("Could not load previous index. Forcing re-processing.")
                     st.session_state.processed_pdf_hash = None
                     st.rerun()

             except Exception as e:
                 logger.error(f"Error re-initializing PDR: {e}", exc_info=True)
                 st.warning("Error during Q&A system re-initialization.")
                 st.session_state.retriever = None

        if st.session_state.current_doc_name != uploaded_file.name and st.session_state.processed_pdf_hash == uploaded_hash:
             st.session_state.current_doc_name = uploaded_file.name
        logger.debug(f"File {uploaded_file.name} (hash: {uploaded_hash}) already processed.")


# --- Setup QA Chain (ConversationalRetrievalChain with PDR) ---
if st.session_state.retriever and st.session_state.llm and not st.session_state.qa_chain:
    logger.info("Attempting to set up ConversationalRetrievalChain...")
    try:
        # Use the revamped system prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(QA_SYSTEM_PROMPT_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=st.session_state.retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
        )
        logger.info("ConversationalRetrievalChain successfully created.")
        st.rerun()

    except Exception as e:
        logger.error(f"Failed to create ConversationalRetrievalChain: {e}", exc_info=True)
        st.error(f"Error setting up the Q&A system: {e}. Check logs.")
        st.session_state.qa_chain = None

elif uploaded_file and not st.session_state.retriever and st.session_state.llm:
     st.warning("⚠️ Document possibly processed, but retriever setup failed. Check logs.")


# --- Display Chat History ---
for msg_data in st.session_state.chat_history_display:
    with st.chat_message(msg_data["role"]):
        st.markdown(msg_data["content"])
        if msg_data["role"] == "assistant" and "sources" in msg_data and msg_data["sources"]:
             with st.expander("View Sources Used"):
                 for i, doc in enumerate(msg_data["sources"]):
                     page_num = doc.metadata.get('page', 'N/A')
                     parent_id = doc.metadata.get('doc_id', 'N/A')
                     source_info = f"Source {i+1} (Page: {page_num}, Parent ID: {parent_id})"
                     preview_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                     st.info(f"{source_info}:\n```\n{preview_content}\n```")


# --- Q&A Interaction Logic ---
if st.session_state.qa_chain:
    if user_input := st.chat_input(f"Ask about '{st.session_state.current_doc_name}'..."):
        st.session_state.chat_history_display.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        # --- <<< INSERT TEMPORARY DIRECT RETRIEVAL TEST HERE >>> ---
        st.info("Running DIRECT RETRIEVAL TEST...") # Display message in UI
        try:
            # Check if the retriever object exists in session state
            if st.session_state.retriever:
                 logger.info(f"Testing retriever directly for query: '{user_input}'")
                 # Use the retriever's standard method to get relevant documents
                 # This bypasses the conversational chain's question processing
                 raw_retrieved_docs = st.session_state.retriever.get_relevant_documents(user_input)
                 # Display the results in an expander in the Streamlit UI
                 with st.expander("DEBUG: Direct PDR Output", expanded=True):
                     st.write(f"Retrieved {len(raw_retrieved_docs)} parent documents directly from PDR:")
                     if raw_retrieved_docs:
                         for i, doc in enumerate(raw_retrieved_docs):
                             page_num = doc.metadata.get('page', 'N/A')
                             parent_id = doc.metadata.get('doc_id', 'N/A') # PDR might add this
                             # Show the metadata and the first 500 chars of the retrieved parent chunk
                             st.info(f"Doc {i+1} (Metadata: Page={page_num}, ID={parent_id})\n```\n{doc.page_content[:500]}...\n```")
                     else:
                          st.write("PDR returned no documents for this query.")
            else:
                 # If retriever hasn't been set up yet
                 st.warning("Retriever object not found in session state for direct test.")
        except Exception as e:
            # Catch errors specifically from the direct retriever call
            st.error(f"Error during direct retriever test: {e}")
            logger.error(f"Error in direct retriever test: {e}", exc_info=True)
        st.info("--- END DIRECT RETRIEVAL TEST ---")
        # --- <<< END TEMPORARY DEBUG BLOCK >>> ---

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            source_documents = []
            start_time = time.time()
            with st.spinner("🤔 Thinking, Retrieving & Synthesizing..."):
                try:
                    logger.info(f"Invoking ConversationalRetrievalChain for query: '{user_input}'")
                    response = st.session_state.qa_chain({"question": user_input})
                    end_time = time.time()
                    result_text = response.get('answer', "Error: Could not retrieve answer.")
                    source_documents = response.get('source_documents', [])
                    logger.info(f"Chain execution time: {end_time - start_time:.2f}s | Docs retrieved: {len(source_documents)}")
                except Exception as e:
                    end_time = time.time()
                    logger.error(f"Error during QA chain execution ({end_time - start_time:.2f}s): {e}", exc_info=True)
                    result_text = f"⚠️ Sorry, an error occurred: {e}"
                    message_placeholder.error(result_text)

            if 'response' in locals(): message_placeholder.markdown(result_text)
            assistant_message_display = {"role": "assistant", "content": result_text, "sources": source_documents}
            st.session_state.chat_history_display.append(assistant_message_display)

            if source_documents:
                 with st.expander("View Sources Used"):
                     for i, doc in enumerate(assistant_message_display["sources"]):
                         page_num = doc.metadata.get('page', 'N/A')
                         parent_id = doc.metadata.get('doc_id', 'N/A')
                         source_info = f"Source {i+1} (Page: {page_num}, Parent ID: {parent_id})"
                         preview_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                         st.info(f"{source_info}:\n```\n{preview_content}\n```")

# --- Handling Initial/Error States ---
elif not uploaded_file:
    st.info("✨ Upload a PDF document to begin asking questions.")
    if st.session_state.processed_pdf_hash:
        logger.info("No file uploaded currently, resetting previous state.")
        for key in ['retriever', 'qa_chain', 'processed_pdf_path', 'processed_pdf_hash', 'current_doc_name', 'chat_history_display', 'memory', 'docstore']:
            if key in st.session_state:
                if key == 'chat_history_display': st.session_state[key] = []
                elif key == 'memory' and hasattr(st.session_state.get(key), 'clear'): st.session_state.get(key).clear()
                elif key == 'docstore' and hasattr(st.session_state.get(key), 'clear'): st.session_state.get(key).clear()
                elif key == 'current_doc_name': st.session_state[key] = "No document loaded"
                else: st.session_state[key] = None
        st.rerun()
elif st.session_state.processed_pdf_path and not st.session_state.qa_chain:
     st.warning("⚠️ Document processed, but Q&A system is not ready. Check logs (API Key/LLM/Retriever setup).")