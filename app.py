import streamlit as st
import fitz
import faiss
import numpy as np
import io
import json
import random
from typing import List, Dict, Tuple

# NLP & models
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Optional heavy features
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    import networkx as nx
    from pyvis.network import Network
    GRAPH_AVAILABLE = True
except Exception:
    GRAPH_AVAILABLE = False

# -------------------------
# Config & Helpers
# -------------------------
st.set_page_config(page_title="StudyMate — Ultimate RAG Tutor", page_icon="📘", layout="wide")
st.title("📘 StudyMate — Ultimate RAG Tutor")
st.caption("RAG + Bloom levels + Why-am-I-wrong + Flashcards/Quiz + OCR + Concept Graph")

EMBED_DIM = 384

@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

EMBEDDER, GENERATOR = load_models()

# -------------------------
# PDF extraction & OCR
# -------------------------
def extract_text_from_pdf_file(file) -> List[Tuple[int,str]]:
    # returns list of (page_no, text)
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
    except Exception:
        return []
    pages = []
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text")
        if not txt and OCR_AVAILABLE:
            # try render page to image and OCR
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = pytesseract.image_to_string(img)
        pages.append((i, txt or ""))
    file.seek(0)
    return pages


def chunk_text(text: str, chunk_words: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap
    return chunks

# Build corpus -> records + FAISS index

def build_corpus(files) -> Tuple[List[Dict], np.ndarray, faiss.IndexFlatL2]:
    records = []
    all_chunks = []
    for f in files:
        pages = extract_text_from_pdf_file(f)
        for page_no, text in pages:
            for ch in chunk_text(text):
                if ch.strip():
                    records.append({"text": ch, "page": page_no, "source": f.name})
                    all_chunks.append(ch)
    if not records:
        index = faiss.IndexFlatL2(EMBED_DIM)
        return [], np.empty((0, EMBED_DIM), dtype="float32"), index
    embeddings = EMBEDDER.encode(all_chunks, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)
    return records, embeddings, index

# search

def search_index(records, index, query: str, k: int = 4):
    if not records:
        return []
    qv = EMBEDDER.encode([query]).astype("float32")
    D, I = index.search(qv, k)
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(records):
            continue
        rec = records[idx]
        # convert L2 to confidence heuristic
        conf = float(np.exp(-dist))
        hits.append({"text": rec["text"], "page": rec["page"], "source": rec["source"], "distance": float(dist), "conf_raw": conf})
    # normalize conf
    if hits:
        s = sum(h["conf_raw"] for h in hits)
        if s>0:
            for h in hits:
                h["confidence"] = round(h["conf_raw"]/s,3)
        else:
            for h in hits:
                h["confidence"] = 0.0
    return hits

# build prompt & answer

def build_context(hits, max_chars=1600):
    parts = []
    cites = []
    used = 0
    for h in hits:
        t = h["text"].strip()
        if used + len(t) > max_chars:
            break
        parts.append(t)
        cites.append(f"{h['source']} · p.{h['page']} (conf {h['confidence']})")
        used += len(t)
    return "\n\n".join(parts), cites


def generate_answer(context: str, question: str, style: str = "neutral") -> str:
    style_map = {
        "neutral": "Answer concisely using only the provided context.",
        "eli5": "Explain like I'm 12 using simple words and an example.",
        "exam": "Answer formally in exam style with key points and definition.",
        "prof": "Be precise and technical; include assumptions if needed.",
    }
    inst = style_map.get(style, style_map["neutral"]) 
    prompt = f"You are an academic assistant. {inst}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    out = GENERATOR(prompt, max_length=320, do_sample=False)
    return out[0]["generated_text"].strip()

# Why am I wrong

def critique_student_answer(student_answer: str, context: str) -> str:
    prompt = (
        "You are an academic tutor. Given the reference context and a student's answer, provide a clear critique:\n"
        "- State correctness points, missing points, and incorrect claims.\n- Suggest a corrected answer.\n\n"
        f"Context:\n{context}\n\nStudent Answer:\n{student_answer}\n\nCritique:"
    )
    out = GENERATOR(prompt, max_length=400, do_sample=False)
    return out[0]["generated_text"].strip()

# Flashcards

def generate_flashcards_from_pool(pool: List[str], n: int = 8) -> List[Dict]:
    joined = "\n\n".join(random.sample(pool, k=min(len(pool), max(1,n*2))))
    prompt = (
        "Generate up to " + str(n) + " concise study flashcards as JSON list like [{\"q\":...,\"a\":...}] from the context."
        " Keep answers short.\n\nContext:\n" + joined + "\n\nFlashcards:"
    )
    out = GENERATOR(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    try:
        data = json.loads(out)
        cards = [{"q": c.get("q",""), "a": c.get("a","")} for c in data if isinstance(c, dict)]
        return cards[:n]
    except Exception:
        # fallback simple split
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        cards = []
        for ln in lines:
            if ln.lower().startswith("q:"):
                q = ln.split(":",1)[1].strip()
                continue
        return []

# Quiz

def generate_quiz_from_pool(pool: List[str], n_q: int = 5) -> str:
    joined = "\n\n".join(random.sample(pool, k=min(len(pool), 5)))
    prompt = (
        f"Create a multiple-choice quiz of {n_q} questions based only on the context. Provide options A-D and mark the correct option after each question.\n\nContext:\n" + joined + "\n\nQuiz:"
    )
    return GENERATOR(prompt, max_length=900, do_sample=False)[0]["generated_text"].strip()

# Concept extraction for graph (simple noun-phrase via embeddings clustering)
def build_concept_graph(records, top_k=40):
    if not GRAPH_AVAILABLE:
        return None
    texts = [r['text'] for r in records]
    # naive: pick top_k distinct short phrases by frequency
    words = ' '.join(texts).split()
    freq = {}
    for w in words:
        w2 = ''.join(ch for ch in w.lower() if ch.isalnum())
        if len(w2)>4:
            freq[w2] = freq.get(w2,0)+1
    items = sorted(freq.items(), key=lambda x:-x[1])[:top_k]
    nodes = [it[0] for it in items]
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    # connect if co-occur in same chunk
    for r in records:
        text = r['text'].lower()
        present = [n for n in nodes if n in text]
        for i in range(len(present)):
            for j in range(i+1,len(present)):
                G.add_edge(present[i], present[j])
    return G

# -------------------------
# Session state init
# -------------------------
if 'records' not in st.session_state:
    st.session_state.records = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("📂 Upload & Settings")
    files = st.file_uploader("Upload PDFs (multi)", type=['pdf'], accept_multiple_files=True)
    chunk_words = st.number_input("Chunk size (words)", min_value=200, max_value=1200, value=500, step=50)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=500, value=100, step=25)
    if st.button("🔁 Build/Refresh Index"):
        if not files:
            st.warning("Upload PDFs first")
        else:
            with st.spinner("Indexing..."):
                st.session_state.records, emb, idx = build_corpus(files)
                st.session_state.index = idx
                st.success(f"Indexed {len(st.session_state.records)} chunks from {len(files)} files.")

    st.markdown('---')
    st.header("Modes & Extras")
    mode = st.radio("Mode", ['Chat','Why I am Wrong','Flashcards','Quiz','Summarize','Concept Graph'])
    style = st.selectbox("Answer Style (Bloom) ", ['remember','understand','apply','analyze','evaluate','create'])
    st.markdown('---')
    if OCR_AVAILABLE:
        st.success("OCR available")
    else:
        st.info("OCR not available. To enable, install pytesseract + pillow and system tesseract.")
    if GRAPH_AVAILABLE:
        st.success("Graph visualization enabled")
    else:
        st.info("Graph libs missing. To enable, install networkx + pyvis")

# -------------------------
# Main
# -------------------------
if not st.session_state.records:
    st.info("Upload PDFs and click Build/Refresh Index to start.")
    if files and not st.session_state.records:
        # auto-build small convenience
        with st.spinner("Indexing uploaded files..."):
            st.session_state.records, emb, idx = build_corpus(files)
            st.session_state.index = idx
            st.success(f"Indexed {len(st.session_state.records)} chunks.")

if not st.session_state.records:
    st.stop()

records = st.session_state.records
index = st.session_state.index

# Helper to map Bloom style to prompt instruction
bloom_map = {
    'remember': 'Give a short definition and key points.',
    'understand': 'Explain clearly with a simple example (ELI5).',
    'apply': 'Show a short worked example applying the concept.',
    'analyze': 'Compare and contrast with related concepts.',
    'evaluate': 'Give pros/cons and critical view.',
    'create': 'Suggest a project or original application using this concept.'
}

# -------------------------
# Modes implementation
# -------------------------
if mode == 'Chat':
    st.subheader('💬 Chat (RAG)')
    for turn in st.session_state.history:
        if turn['role']=='user':
            st.chat_message('user').markdown(turn['content'])
        else:
            st.chat_message('assistant').markdown(turn['content'])
            if 'cites' in turn:
                with st.expander('🔍 References & confidence'):
                    for c in turn['cites']:
                        st.write('-',c)

    prompt = st.chat_input('Ask something from your PDFs...')
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.history.append({'role':'user','content':prompt})
        hits = search_index(records,index,prompt,k=4)
        ctx, cites = build_context(hits)
        if not ctx:
            ans = "No relevant content found in the uploaded PDFs."
            st.chat_message('assistant').markdown(ans)
            st.session_state.history.append({'role':'assistant','content':ans})
        else:
            extra = bloom_map.get(style,'')
            answer = generate_answer(ctx + "\n\nInstruction: " + extra, prompt, style='neutral')
            # overall confidence (average)
            avg_conf = round(sum(h['confidence'] for h in hits)/len(hits),3) if hits else 0.0
            display = f"**Answer (confidence {avg_conf*100:.0f}%)**\n\n" + answer
            with st.chat_message('assistant'):
                st.markdown(display)
                with st.expander('🔍 References & snippets'):
                    for h in hits:
                        st.write(f"- {h['source']} · p.{h['page']} (conf {h['confidence']})")
                        st.caption(h['text'][:300] + ("…" if len(h['text'])>300 else ""))
            st.session_state.history.append({'role':'assistant','content':display,'cites':[f"{h['source']} · p.{h['page']} (conf {h['confidence']})" for h in hits]})

elif mode == 'Why I am Wrong':
    st.subheader('📝 Why Am I Wrong? — Student self-assessment')
    student_ans = st.text_area('Paste your answer here (what you wrote):', height=200)
    question = st.text_input('Related question (optional, helps retrieval):')
    if st.button('🔎 Critique my answer'):
        if not student_ans:
            st.warning('Paste your answer first')
        else:
            if question:
                hits = search_index(records,index,question,k=4)
                ctx,_ = build_context(hits)
            else:
                # use entire doc pool (sample)
                pool = [r['text'] for r in random.sample(records, k=min(10,len(records)))]
                ctx = '\n\n'.join(pool)
            critique = critique_student_answer(student_ans, ctx)
            st.markdown('**Tutor critique:**')
            st.write(critique)
            # also produce corrected short answer
            corrected = generate_answer(ctx, question or student_ans, style='prof')
            st.markdown('**Suggested corrected answer:**')
            st.write(corrected)

elif mode == 'Flashcards':
    st.subheader('🗂 Auto flashcards from your PDFs')
    pool = [r['text'] for r in random.sample(records, k=min(40,len(records)))]
    n = st.slider('How many flashcards?', 3, 30, 8)
    if st.button('Generate Flashcards'):
        cards = generate_flashcards_from_pool(pool, n=n)
        if not cards:
            st.warning('Could not generate flashcards. Try again or increase pool size.')
        else:
            for i,c in enumerate(cards, start=1):
                with st.expander(f"Card {i}: {c['q']}"):
                    st.markdown(c['a'])
            st.download_button('Download Flashcards (JSON)', json.dumps(cards, indent=2), file_name='flashcards.json')

elif mode == 'Quiz':
    st.subheader('📝 Generate MCQ Quiz')
    n_q = st.slider('Number of questions', 3, 20, 5)
    pool = [r['text'] for r in random.sample(records, k=min(40,len(records)))]
    if st.button('Generate Quiz'):
        quiz = generate_quiz_from_pool(pool, n_q=n_q)
        st.text_area('Quiz (copyable)', quiz, height=420)
        st.download_button('Download Quiz (TXT)', quiz, file_name='quiz.txt')

elif mode == 'Summarize':
    st.subheader('🧾 Summarize a document')
    sources = sorted({r['source'] for r in records})
    which = st.selectbox('Pick a source file', sources)
    scope = st.slider('How many chunks to include', 1, 50, 10)
    target = [r['text'] for r in records if r['source']==which][:scope]
    ctx = '\n\n'.join(target)
    prompt = f"Summarize the key ideas for quick revision. Style: {style}."
    if st.button('Generate Summary'):
        s = GENERATOR(prompt + "\n\nContext:\n" + ctx, max_length=600, do_sample=False)[0]['generated_text']
        st.markdown(s)
        st.download_button('Download Summary', s, file_name='summary.txt')

elif mode == 'Concept Graph':
    st.subheader('🧠 Concept Graph (auto)')
    st.info('Graphing requires networkx + pyvis installed on server.')
    if not GRAPH_AVAILABLE:
        st.warning('Graph libs are not available in this environment.')
    else:
        with st.spinner('Building graph...'):
            G = build_concept_graph(records, top_k=40)
            if G is None or len(G.nodes)==0:
                st.warning('Not enough content for graph.')
            else:
                net = Network(height='600px', width='100%', notebook=False)
                net.from_nx(G)
                net.repulsion(node_distance=200)
                path = 'graph.html'
                net.show(path)
                with open(path,'r',encoding='utf-8') as f:
                    html = f.read()
                st.components.v1.html(html, height=650, scrolling=True)

# Footer
st.markdown('---')
st.caption('StudyMate — RAG tutor. For heavy usage consider running locally with larger models.\nTips: If OCR or Graph missing, install pytesseract, pillow, tesseract-ocr, networkx, pyvis.')
