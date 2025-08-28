import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)
chunks, embeddings = [], []

# HuggingFace text generation model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# PDF text extraction
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Chunk text
def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# --- Streamlit UI ---
st.set_page_config(page_title="StudyMate", page_icon="📘", layout="wide")
st.title("📘 StudyMate - AI PDF Q&A Assistant")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# Sidebar for PDF upload
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        chunks.clear()
        embeddings.clear()
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            cks = chunk_text(text)
            chunks.extend(cks)
            emb = embedder.encode(cks)
            embeddings.extend(emb)
        embeddings_np = np.array(embeddings).astype("float32")
        index.reset()
        index.add(embeddings_np)
        st.session_state.index_ready = True
        st.success("✅ PDFs processed and ready!")

# Display chat history
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)

# Chat input
if st.session_state.index_ready:
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.history.append(("user", prompt))

        # Search relevant chunks
        q_emb = embedder.encode([prompt]).astype("float32")
        D, I = index.search(q_emb, k=3)
        context = " ".join([chunks[i] for i in I[0]])

        # Generate answer
        query = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
        response = qa_pipeline(query, max_length=256, do_sample=False)
        answer = response[0]["generated_text"]

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("🔍 References"):
                for i in I[0]:
                    st.write(f"- {chunks[i][:200]}...")

        # Save to history
        st.session_state.history.append(("assistant", answer))
else:
    st.info("👆 Upload PDFs in the sidebar to start asking questions.")
