import streamlit as st
import logging
import os
import io
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import faiss
import pickle
import json
import numpy as np
import re

# --- In-memory log capture for Streamlit ---
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# --- File-based log capture ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "chatbot_logs.log")

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# --- Configure logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid adding handlers multiple times if Streamlit reruns
if not logger.hasHandlers():
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

logger.propagate = False

# --- Load Models Safely ---
@st.cache_resource
def load_models():
    try:
        logger.info("🔄 Loading FAISS index and metadata...")
        faiss_index = faiss.read_index("faiss_index_bge_base.index")
        with open("faiss_metadata.pkl", "rb") as f:
            passages = pickle.load(f)
        logger.info(f"✅ Loaded {len(passages)} passages.")

        logger.info("🔄 Loading embedding model on CPU...")
        embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        logger.info("✅ Embedding model loaded.")

        logger.info("🔄 Loading Mistral model...")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        logger.info("✅ Mistral model + tokenizer loaded.")


        # Load Zephyr for enrichment (as pipeline)
        logger.info("🔄 Loading Zephyr-7B chat pipeline for offline chunk enrichment...")
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        logger.info("✅ Zephyr chat pipeline loaded.")

        return faiss_index, passages, embedding_model, model, tokenizer, pipe

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# --- UI Configuration ---
st.set_page_config(
    page_title="InsuranceGPT Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #F0F7FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .stTextInput>div>div>input {
        font-size: 1rem;
        padding: 0.75rem;
    }
    .upload-section {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 0.25rem solid #2196F3;
    }
    .bot-message {
        background-color: #F1F8E9;
        border-left: 0.25rem solid #4CAF50;
    }
    .status-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFFDE7;
        margin-bottom: 1rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ECEFF1;
        text-align: center;
        color: #78909C;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models with a loading indicator
with st.spinner("🔄 Loading models and knowledge base..."):
    faiss_index, passages, embedding_model, model, tokenizer, pipe = load_models()

# 🔍 Safety check
if passages is None:
    st.error("❌ 'passages' is None. Check your faiss_metadata.pkl file.")
    st.stop()

if not isinstance(passages, list):
    st.error("❌ 'passages' must be a list. Got: " + str(type(passages)))
    st.stop()

if len(passages) == 0:
    st.warning("⚠️ 'passages' is empty.")

# --- Utility functions (untouched) ---
def clean_text(text):
    return re.sub(r"\W+", " ", text.lower()).split()

def hybrid_search(query, top_k=5, alpha=0.6):
    logger.info(f"🔍 Running hybrid search for: {query}")
    query_vec = embedding_model.encode(query)
    dense_scores, dense_indices = faiss_index.search(query_vec.reshape(1, -1), len(passages))

    query_words = set(clean_text(query))
    keyword_scores = []

    for idx in dense_indices[0]:
        doc = passages[idx]
        meta = doc
        section_text = meta.get("section_title", "") + " " + meta.get("content", "")
        doc_words = set(clean_text(section_text))
        overlap = len(query_words & doc_words)
        keyword_scores.append((idx, overlap))

    dense_scores = dense_scores[0]
    dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-6)
    keyword_vals = np.array([score for _, score in keyword_scores])
    keyword_vals = (keyword_vals - np.min(keyword_vals)) / (np.max(keyword_vals) - np.min(keyword_vals) + 1e-6)

    combined_scores = alpha * dense_scores + (1 - alpha) * keyword_vals
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        doc = passages[dense_indices[0][i]]
        meta = doc
        results.append({
            "content": meta.get("content", ""),
            "policy_type": meta.get("policy_type", ""),
            "coverage": meta.get("coverage", "")
        })

    logger.info(f"✅ Retrieved {len(results)} top sections for query: {query}")
    return results


def build_prompt(context: str, query: str) -> str:
    return f"""
You are a helpful and knowledgeable assistant specialized in insurance policies. Based on the provided context, respond clearly and politely to the user's question. Your answer should be accurate, easy to understand, and based only on the given context.

If you cannot find a reliable answer from the context, simply respond with:
"Sorry to say, I don't know the answer for this question."

Context:
{context}

User Question:
{query}

Chatbot Answer:"""

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing_query" not in st.session_state:
    st.session_state.processing_query = False

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# --- UI Layout ---
# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h1 class="main-header">🛡️ InsuranceGPT Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        Ask questions about your insurance policies and get accurate answers powered by AI. 
        Upload new policies using the sidebar to expand the knowledge base.
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <b>You:</b> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <b>InsuranceGPT:</b> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Process user query - FIX: Use a form to prevent auto-rerun issues
    with st.form(key="query_form"):
        user_input = st.text_input(
            "Ask a question about your insurance policy:",
            placeholder="Example: What is my deductible for earthquake damage?",
            key="user_query"
        )
        submit_button = st.form_submit_button("Send")

    # Only process when the form is submitted
    if submit_button and user_input and user_input != st.session_state.current_query:
        st.session_state.current_query = user_input
        st.session_state.processing_query = True
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        try:
            with st.spinner("🤔 Analyzing your question..."):
                context = hybrid_search(user_input)
                prompt = build_prompt(context, user_input)

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(inputs.input_ids, max_new_tokens=256, do_sample=False)
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the part after "Chatbot Answer:"
                match = re.search(r"Chatbot Answer:\s*(.*)", full_response, re.DOTALL)
                response = match.group(1).strip() if match else full_response.strip()
                
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error("⚠️ Something went wrong while generating a response.")
            logger.error(f"Response error: {e}")
        
        # Reset the processing flag
        st.session_state.processing_query = False
        # Rerun once to update the UI with new messages
        st.experimental_rerun()

with col2:
    st.markdown('<h3 class="sub-header">📊 System Status</h3>', unsafe_allow_html=True)
    
    # System status indicators
    st.markdown("""
    <div class="status-container">
        <p><b>🧠 Models:</b> Loaded and running</p>
        <p><b>📚 Knowledge Base:</b> Active</p>
        <p><b>🔍 Search Engine:</b> Hybrid FAISS</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show knowledgebase stats
    st.markdown('<h3 class="sub-header">📚 Knowledge Base</h3>', unsafe_allow_html=True)
    
    # Count unique policies
    if passages:
        policy_sources = set()
        for passage in passages:
            if "source" in passage:
                policy_sources.add(passage["source"])
        
        st.markdown(f"""
        <div class="status-container">
            <p><b>Total Passages:</b> {len(passages)}</p>
            <p><b>Unique Policies:</b> {len(policy_sources)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tips section
    st.markdown('<h3 class="sub-header">💡 Tips</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="status-container">
        <p>• Ask specific questions about your policy coverage</p>
        <p>• Include relevant details like coverage type</p>
        <p>• Upload new policies via the sidebar</p>
    </div>
    """, unsafe_allow_html=True)

# --- PDF Upload Section (Sidebar) ---
with st.sidebar:
    st.markdown('<h2 class="sub-header">➕ Add New Policy</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-section">
        Upload an insurance policy PDF to add it to the knowledge base. 
        The system will automatically extract and index the content.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select PDF file", type=["pdf"], key="pdf_uploader")

    # Initialize session state variables
    if "upload_processed" not in st.session_state:
        st.session_state.upload_processed = False
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = ""

    # If a new file is uploaded, reset flags
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.upload_processed = False
        st.session_state.last_uploaded_filename = uploaded_file.name

    if uploaded_file and not st.session_state.upload_processed:
        logger.info("📤 Uploaded file received and not yet processed.")
        
        with st.spinner("Processing your policy..."):
            os.makedirs("policy_pdfs", exist_ok=True)
            pdf_save_path = os.path.join("policy_pdfs", uploaded_file.name)

            # Save the file
            file_content = uploaded_file.read()
            with open(pdf_save_path, "wb") as f:
                f.write(file_content)

            st.success(f"✅ File saved: {uploaded_file.name}")
            logger.info(f"✅ Saved uploaded file to: {pdf_save_path}")

            # Run the pipeline
            from pathlib import Path
            import time
            from knowledge_update import (
                extract_text_from_pdf,
                chunk_text_by_paragraphs,
                enrich_chunk_with_zephyr
            )

            progress_bar = st.progress(0)
            st.markdown("🔄 Extracting text from PDF...")
            text = extract_text_from_pdf(pdf_save_path)
            progress_bar.progress(25)
            
            st.markdown("🔄 Chunking into sections...")
            chunks = chunk_text_by_paragraphs(text)
            progress_bar.progress(50)
            
            st.markdown("🔄 Enriching content with AI...")
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                enriched_chunk = enrich_chunk_with_zephyr(
                    section_text=chunk["content"],
                    section_title=chunk["section_title"],
                    source=uploaded_file.name,
                    pipe=pipe
                )
                enriched_chunks.append(enriched_chunk)
                if len(chunks) > 0:
                    progress_portion = 25 * (i + 1) / len(chunks)
                    progress_bar.progress(50 + int(progress_portion))

            os.makedirs("./enriched_jsons", exist_ok=True)
            enriched_json_path = f"./enriched_jsons/{Path(uploaded_file.name).stem}_enriched.json"
            with open(enriched_json_path, "w", encoding="utf-8") as f:
                json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
            
            st.markdown("🧠 Updating knowledge base...")
            progress_bar.progress(85)

            for chunk in enriched_chunks:
                text = chunk["text"]
                metadata = chunk["metadata"]
                metadata.pop("error", None)
                vector = embedding_model.encode(text, show_progress_bar=False)
                faiss_index.add(np.array([vector]).astype("float32"))
                passages.append(metadata)

            # Save updated FAISS + metadata
            faiss.write_index(faiss_index, "faiss_index_bge_base.index")
            with open("faiss_metadata.pkl", "wb") as f:
                pickle.dump(passages, f)
            
            progress_bar.progress(100)
            st.success("✅ PDF successfully added to knowledge base!")

            # ✅ Mark as processed and rerun to reset uploader
            st.session_state.upload_processed = True
            time.sleep(1)
            st.experimental_rerun()
    
    # Show upload history if available
    if st.session_state.get("last_uploaded_filename", ""):
        st.markdown('<h3 class="sub-header">📚 Recently Added</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="status-container">
            <p>• {st.session_state.last_uploaded_filename}</p>
        </div>
        """, unsafe_allow_html=True)

    # Clear chat button in a form to prevent infinite reruns
    with st.form(key="clear_form"):
        clear_button = st.form_submit_button("🗑️ Clear Chat History")
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.current_query = ""
            # No rerun needed here as the form submit will handle it

# Footer
st.markdown("""
<div class="footer">
    Powered by Mistral 7B + Hybrid FAISS Search | © 2025 InsuranceGPT Assistant
</div>
""", unsafe_allow_html=True)