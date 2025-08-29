import os
import re
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from dotenv import load_dotenv, set_key, find_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"
PROMPT_TEMPLATE = """
You are a helpful assistant. Use the provided context to answer the query. If you don't know the answer, state that you don't know. Be concise and factual.

Query: {user_query}

Context: {document_context}

Answer:
"""

# Load environment variables from .env (persisted across sessions)
ENV_PATH = find_dotenv(usecwd=True) or ".env"
if not os.path.exists(ENV_PATH):
    # Create an empty .env if missing so we can persist keys later
    open(ENV_PATH, "a").close()
load_dotenv(dotenv_path=ENV_PATH, override=False)
# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    os.makedirs("document_store/pdfs", exist_ok=True)
    file_path = f"document_store/pdfs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to load the PDF document
def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

# Function to split the document into chunks
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    # Filter out empty or whitespace-only chunks to avoid 400 errors from embedding APIs
    filtered = [
        doc for doc in chunks
        if getattr(doc, "page_content", None) and doc.page_content.strip()
    ]
    return filtered

# Function to create an index of the document chunks
def index_documents(chunks):
    # Try bulk add first; on error, fall back to per-chunk to skip bad ones
    try:
        Document_Vector_db.add_documents(chunks)
    except Exception:
        valid_chunks = []
        for doc in chunks:
            try:
                if doc.page_content and doc.page_content.strip():
                    Document_Vector_db.add_documents([doc])
                    valid_chunks.append(doc)
            except Exception:
                # Skip problematic chunk
                continue
        return valid_chunks

# Function to find related documents based on a query
def find_related_documents(query):
    return Document_Vector_db.similarity_search(query)

# Function to generate an answer
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = prompt | Language_Model
    try:
        result = response_chain.invoke({"user_query": query, "document_context": context})
    except ResourceExhausted as e:
        # 429 quota error: fallback to Flash once if not already
        try:
            fallback_model = ChatGoogleGenerativeAI(
                model=GEMINI_DEFAULT_MODEL,
                google_api_key=os.getenv("GOOGLE_API_KEY") or st.session_state.get("GOOGLE_API_KEY"),
                temperature=st.session_state.get("temperature", 0.3) if hasattr(st, "session_state") else 0.3,
            )
            fallback_chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | fallback_model
            result = fallback_chain.invoke({"user_query": query, "document_context": context})
        except Exception:
            raise RuntimeError(
                "Gemini quota exceeded (429). Fallback to flash also failed. Reduce requests or wait for quota reset."
            ) from e
    except Exception as e:
        raise RuntimeError(
            "Model call failed. If using Gemini, verify your API key is valid and not restricted, and that the model is enabled for the key."
        ) from e
    raw_text = result.content if hasattr(result, "content") else str(result)
    # Remove DeepSeek R1 reasoning tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE)
    return cleaned.strip()
# Streamlit UI
st.title("RAG application")
st.markdown("Ask from PDF")

# Sidebar: model/provider selection
with st.sidebar:
    st.subheader("Model Settings")
    provider = st.selectbox(
        "Provider",
        [
            "Ollama (Local)",
            "Gemini (Google)",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    # Gemini setup inputs
    # Manage Google API key from env or user input
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.session_state.get("GOOGLE_API_KEY")
    if provider.startswith("Gemini"):
        # No model selector; we force Flash in code to avoid quota issues
        api_input = st.text_input(
            "Google API Key",
            value="",
            type="password",
            help="You can also set environment variable GOOGLE_API_KEY.",
        )
        save_key = st.checkbox("Save key to .env for future sessions", value=False)
        if api_input:
            st.session_state["GOOGLE_API_KEY"] = api_input
            google_api_key = api_input
            if save_key:
                try:
                    set_key(ENV_PATH, "GOOGLE_API_KEY", api_input)
                    os.environ["GOOGLE_API_KEY"] = api_input
                    st.success(f"Saved GOOGLE_API_KEY to {ENV_PATH}")
                except Exception as e:
                    st.warning(f"Could not save key to {ENV_PATH}: {e}")
        
        with st.expander("Validate Gemini API key"):
            st.caption("Runs a tiny test call to check if your key works before using RAG.")
            if st.button("Run key validation"):
                if not google_api_key:
                    st.error("No key provided.")
                else:
                    try:
                        genai.configure(api_key=google_api_key)
                        test_model = genai.GenerativeModel("gemini-1.5-flash")
                        _ = test_model.generate_content("ping")
                        st.success("Key looks valid and usable.")
                    except Exception as e:
                        st.error(
                            "Gemini API call failed. Possible causes: invalid/expired key, using a Cloud key instead of AI Studio key, missing API enablement, or IP/domain restrictions."
                        )
                        st.exception(e)


# Initialize models and vector store (default keeps current behavior)
if provider.startswith("Gemini"):
    if not google_api_key:
        st.error("Please provide a Google API Key to use Gemini.")
        st.stop()
    try:
        Embedding_Model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=google_api_key
        )
    except Exception:
        try:
            Embedding_Model = GoogleGenerativeAIEmbeddings(
                model="text-embedding-004", google_api_key=google_api_key
            )
        except Exception as e:
            st.error("Failed to initialize embeddings with your key.")
            st.exception(e)
            st.stop()
    try:
        # Force default to Gemini Flash to avoid free-tier quota issues
        Language_Model = ChatGoogleGenerativeAI(
            model=GEMINI_DEFAULT_MODEL,
            google_api_key=google_api_key,
            temperature=temperature,
        )
    except Exception as e:
        st.error("Failed to initialize Gemini chat model with your key.")
        st.exception(e)
        st.stop()
else:
    Embedding_Model = OllamaEmbeddings(model="nomic-embed-text")
    Language_Model = ChatOllama(model="deepseek-r1:1.5b", temperature=temperature)

Document_Vector_db = InMemoryVectorStore(Embedding_Model)

# File uploader
uploaded_PDF = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_PDF:
    file_path = save_uploaded_file(uploaded_PDF)
    raw_docs = load_pdf_documents(file_path)
    chunks = chunk_documents(raw_docs)
    indexed = index_documents(chunks)
    if isinstance(indexed, list):
        st.info(f"Indexed {len(indexed)} chunks after filtering/validation")
    st.success("Document Processed Successfully")

    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("Analyzing Documents....."):
            related_docs = find_related_documents(user_query)
            answer = generate_answer(user_query, related_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(answer)
            