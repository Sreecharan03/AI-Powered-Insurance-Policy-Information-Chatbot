# 🧠 AI-Powered Insurance Policy Information Chatbot

A dynamic Retrieval-Augmented Generation (RAG)-based chatbot for real-time, document-aware responses to insurance policy queries. Built with hybrid search, structured content extraction, and LLM-powered generation to deliver accurate, conversational answers from uploaded PDFs.

## 🚀 Overview

This chatbot is designed to transform static insurance PDFs into a searchable knowledge base and respond intelligently to customer queries. It supports real-time document ingestion, structured metadata extraction, and semantic + keyword-based hybrid search.

---

## 🧩 Features

- 🔍 **Hybrid Retrieval (Dense + Sparse Search)**
- 📄 **Runtime PDF Uploads with Auto-Indexing**
- 🧾 **LLM-Based Metadata Structuring (Zephyr-7B)**
- 🧠 **Response Generation with Mistral-7B-Instruct**
- 🧠 **Section-Aware Chunking with Paragraph Boundaries**
- ⚙️ **FAISS for Scalable Vector Retrieval**
- 🔁 **Fallback Mechanisms (Regex-based)**
- 📊 **Real-world policy support (e.g., HDFC home & health)**

---

## 🛠️ Tech Stack

| Component         | Technology                                |
|------------------|-------------------------------------------|
| Embeddings        | `BAAI/bge-base-en-v1.5` (768d)             |
| LLM (QA)          | `Mistral-7B-Instruct-v0.2`                |
| LLM (Structuring) | `Zephyr-7B-beta`                          |
| PDF Parser        | `PyMuPDF` (`fitz`)                        |
| Vector DB         | `FAISS` (CPU-optimized index)             |
| UI Framework      | `Streamlit`                               |

---

## ⚙️ System Architecture

1. **Document Ingestion**
   - PDF parsing using PyMuPDF
   - Paragraph-based chunking
   - Unique section identification

2. **Metadata Structuring**
   - Zephyr LLM extracts coverage, classification, and key content
   - Stored as structured JSON

3. **Vector Storage**
   - Embeddings generated per chunk
   - Stored and queried using FAISS

4. **Hybrid Search**
   - Combines semantic similarity and keyword overlap
   - Top-k results with reranking and scoring

5. **Query Response**
   - Mistral LLM forms structured, conversational answers
   - Output formatted for readability

6. **Knowledge Base Expansion**
   - Upload new PDFs via UI
   - Indexed in real-time, deduplicated automatically

---

## 🏗️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Sreecharan03/AI-Powered-Insurance-Policy-Information-Chatbot.git
cd insurance-chatbot

# Create virtual environment and activate
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py
