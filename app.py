# app.py
# StudyMate — Free local RAG tutor (Streamlit + HF local models)
# Features: multi-PDF upload, RAG (FAISS + MiniLM), HF generation (flan-t5-small), 
# Bloom styles, "Why am I wrong", flashcards, quiz, OCR fallback, STT/TTS (local)
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import io, json, os, random, tempfile, time
from typing import List, Dict, Tuple

# Models
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Optional: OCR & STT/TTS
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# faster-whisper (local whisper) optional
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# pyttsx3 TTS (offline)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

st.set_page_config(page_title="StudyMate (Free)", layout="wide")
st.title("📘 StudyMate — Free local RAG tutor (no paid APIs)")

# ---------------------------
# Load lightweight models (cache)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # Embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast (384 dim)
    # Generator: light seq2seq model that runs on CPU tolerably
    gen_model_name = "google/flan-t5-small"  # swap to larger if you have power
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    return embedder, tokenizer, model

EMBEDDER, TOKENIZER, GEN_MODEL = load_models()

# create a HF pipeline wrapper for convenience
@st.cache_resource(show_spinner=False)
def gen_pipeline():
    return pipeline("text2text-generation", model=GEN_MODEL, tokenizer=TOKENIZER, device_map="auto" if "CUDA_VISIBLE_DEVICES" in os.environ else None)
GENERATOR = gen_pipeline()

EMBED_DIM = 384

# ---------------------------
# Utilities: extract, chunk, index
# ---------------------------
def extract_pages(file: io.BytesIO) -> List[Tuple[int,str]]:
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
    except Exception:
        return []
    pages = []
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text")
        if not txt and OCR_AVAILABLE:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = pytesseract.image_to_string(img)
        pages.append((i, txt or ""))
    file.seek(0)
    return pages

def chunk_text(text: str, chunk_words: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_words]))
        i += chunk_words - overlap
    return chunks

def build_corpus(files: List[io.BytesIO], chunk_words=400, overlap=80):
    records = []  # dicts: text, page, source
    all_chunks = []
    for f in files:
        pages = extract_pages(f)
        for page_no, txt in pages:
            for ch in chunk_text(txt, chunk_words, overlap):
                if ch.strip():
                    records.append({"text": ch, "page": page_no, "source": getattr(f, "name", "uploaded")})
                    all_chunks.append(ch)
    if not records:
        idx = faiss.IndexFlatL2(EMBED_DIM)
        return records, np.empty((0, EMBED_DIM), dtype="float32"), idx
    embeddings = EMBEDDER.encode(all_chunks, convert_to_numpy=True).astype("float32")
    idx = faiss.IndexFlatL2(EMBED_DIM)
    idx.add(embeddings)
    return records, embeddings, idx

def search_index(records, index, query: str, k: int = 4):
    if not records:
        return []
    qv = EMBEDDER.encode([query]).astype("float32")
    D, I = index.search(qv, k)
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(records): continue
        rec = records[idx]
        conf_raw = float(np.exp(-dist))
        hits.append({"text": rec["text"], "source": rec["source"], "page": rec["page"], "dist": float(dist), "conf_raw": conf_raw})
    # normalize
    total = sum(h["conf_raw"] for h in hits) or 1.0
    for h in hits:
        h["confidence"] = round(h["conf_raw"]/total, 3)
    return hits

def build_context(hits, max_chars=1600):
    parts, cites = [], []
    used = 0
    for h in hits:
        t = h["text"].strip()
        if used + len(t) > max_chars: break
        parts.append(t)
        cites.append(f"{h['source']} · p.{h['page']} (conf {h['confidence']})")
        used += len(t)
    return "\n\n".join(parts), cites

# ---------------------------
# Generation helpers
# ---------------------------
bloom_map = {
    "remember": "Give a short definition and key points.",
    "understand": "Explain simply with a short example (ELI5).",
    "apply": "Show a short worked example applying the concept.",
    "analyze": "Compare and contrast with related concepts.",
    "evaluate": "Give pros/cons and a critical view.",
    "create": "Suggest a small project or original application."
}

def generate_answer_hf(context: str, question: str, style: str = "remember"):
    instr = bloom_map.get(style, "")
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nInstruction: {instr}\nAnswer concisely and only using the context."
    out = GENERATOR(prompt, max_length=256, do_sample=False)
    return out[0]["generated_text"].strip()

def critique_answer(student_ans: str, context: str):
    prompt = (
        "You are an academic tutor. Using ONLY the context, compare the student's answer to the context.\n"
        "List: (1) Correct points, (2) Missing points, (3) Incorrect claims, (4) A corrected concise answer.\n\n"
        f"Context:\n{context}\n\nStudent Answer:\n{student_ans}\n\nCritique:"
    )
    out = GENERATOR(prompt, max_length=320, do_sample=False)
    return out[0]["generated_text"].strip()

def generate_flashcards(pool: List[str], n=8):
    joined = "\n\n".join(random.sample(pool, k=min(len(pool), max(1, n*2))))
    prompt = f"From the context below generate up to {n} flashcards as JSON array of objects like {{\"q\":\"...\",\"a\":\"...\"}}. Keep answers short.\n\n{joined}\n\nFlashcards:"
    out = GENERATOR(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    try:
        cards = json.loads(out)
        if isinstance(cards, list):
            return cards[:n]
    except Exception:
        # fallback: naive Q/A extraction
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        cards = []
        q = None
        for ln in lines:
            if ln.lower().startswith("q:"):
                q = ln.split(":",1)[1].strip()
            elif ln.lower().startswith("a:") and q:
                a = ln.split(":",1)[1].strip()
                cards.append({"q":q,"a":a})
                q = None
        return cards[:n]
    return []

def generate_quiz(pool: List[str], n_q=5):
    joined = "\n\n".join(random.sample(pool, k=min(len(pool), 5)))
    prompt = f"Create {n_q} multiple-choice questions (A-D) based ONLY on the context. For each question include the correct option letter after the options.\n\n{joined}\n\nQuiz:"
    out = GENERATOR(prompt, max_length=800, do_sample=False)[0]["generated_text"]
    return out

# ---------------------------
# STT/TTS helpers (optional)
# ---------------------------
def tts_speak(text: str, filename: str = "tts_out.mp3"):
    if TTS_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, filename)
            engine.runAndWait()
            return filename
        except Exception:
            pass
    # fallback: return None (no TTS)
    return None

def stt_from_file(file_bytes: bytes):
    if not WHISPER_AVAILABLE:
        return None, "Whisper not installed"
    # saves to temp and runs faster-whisper
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    model = WhisperModel("small", device="cpu", compute_type="int8")  # small model
    segments, info = model.transcribe(tmp.name, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    try:
        os.unlink(tmp.name)
    except: pass
    return text, None

# ---------------------------
# Streamlit UI / session state
# ---------------------------
if "records" not in st.session_state:
    st.session_state.records = []
if "index" not in st.session_state:
    st.session_state.index = None
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar — uploads & settings
with st.sidebar:
    st.header("📂 Upload & Settings (free & local)")
    uploaded = st.file_uploader("Upload PDFs (multi)", type=["pdf"], accept_multiple_files=True)
    chunk_words = st.number_input("Chunk size (words)", min_value=200, max_value=1200, value=400, step=50)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=800, value=80, step=20)
    if st.button("🔁 Build/Refresh Index"):
        if not uploaded:
            st.warning("Upload PDFs first")
        else:
            files = []
            for f in uploaded:
                # keep in-memory file-like with name
                bio = io.BytesIO(f.read())
                bio.name = f.name
                files.append(bio)
            with st.spinner("Indexing (this may take a while for large PDFs)..."):
                recs, emb, idx = build_corpus(files, chunk_words=chunk_words, overlap=overlap)
                st.session_state.records = recs
                st.session_state.index = idx
                st.success(f"Indexed {len(recs)} chunks from {len(uploaded)} files")

    st.markdown("---")
    st.header("Modes & Options")
    mode = st.radio("Mode", ["Chat","Why I'm Wrong","Flashcards","Quiz","Summarize","Concepts"])
    style = st.selectbox("Bloom style (Answer depth)", list(bloom_map.keys()), index=1)
    st.markdown("---")
    st.write("Optional extras (local):")
    st.write(f"OCR available: {'✅' if OCR_AVAILABLE else '❌'}")
    st.write(f"Local Whisper STT: {'✅' if WHISPER_AVAILABLE else '❌'}")
    st.write(f"Local TTS: {'✅' if TTS_AVAILABLE else '❌'}")

# If no index, try build automatically if uploaded
if not st.session_state.records:
    if uploaded:
        # auto-build smaller first
        files = []
        for f in uploaded:
            bio = io.BytesIO(f.read()); bio.name = f.name; files.append(bio)
        with st.spinner("Indexing uploaded PDFs..."):
            recs, emb, idx = build_corpus(files, chunk_words=chunk_words, overlap=overlap)
            st.session_state.records = recs
            st.session_state.index = idx
            st.success(f"Indexed {len(recs)} chunks")
    else:
        st.info("Upload PDFs and click 'Build/Refresh Index' to begin.")
        st.stop()

records = st.session_state.records
index = st.session_state.index

# UI per mode
if mode == "Chat":
    st.subheader("💬 Chat (RAG)")
    for turn in st.session_state.history:
        role = turn.get("role")
        content = turn.get("content")
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").markdown(content)
            if "cites" in turn:
                with st.expander("🔍 References & confidence"):
                    for c in turn["cites"]:
                        st.write("-", c)

    prompt = st.chat_input("Ask a question about your uploaded PDFs...")
    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.history.append({"role":"user","content":prompt})
        hits = search_index(records, index, prompt, k=4)
        ctx, cites = build_context(hits, max_chars=1800)
        if not ctx.strip():
            ans = "I couldn't find relevant content in uploaded PDFs. Try rephrasing or adding more PDFs."
            st.chat_message("assistant").write(ans)
            st.session_state.history.append({"role":"assistant","content":ans})
        else:
            # generation
            answer = generate_answer_hf(ctx, prompt, style=style)
            avg_conf = round(sum(h["confidence"] for h in hits)/len(hits),3) if hits else 0.0
            display = f"**Answer (confidence {avg_conf*100:.0f}%):**\n\n{answer}"
            with st.chat_message("assistant"):
                st.markdown(display)
                with st.expander("🔍 References & snippets"):
                    for h in hits:
                        st.write(f"- {h['source']} · p.{h['page']} (conf {h['confidence']})")
                        st.caption(h['text'][:280] + ("…" if len(h['text'])>280 else ""))
            st.session_state.history.append({"role":"assistant","content":display,"cites":cites})

elif mode == "Why I'm Wrong":
    st.subheader("📝 Why I'm Wrong — paste your attempt and get critique")
    question = st.text_input("Related question (optional)")
    student_ans = st.text_area("Paste your answer here (what you wrote):", height=200)
    if st.button("🔎 Critique my answer"):
        if not student_ans.strip():
            st.warning("Please paste your answer.")
        else:
            if question.strip():
                hits = search_index(records, index, question, k=6)
                ctx, cites = build_context(hits, max_chars=2000)
            else:
                # sample a few chunks
                pool = [r["text"] for r in random.sample(records, k=min(10,len(records)))]
                ctx = "\n\n".join(pool)
            critique = critique_answer(student_ans, ctx)
            st.markdown("**Tutor critique:**")
            st.write(critique)
            st.markdown("**Suggested corrected concise answer:**")
            corrected = generate_answer_hf(ctx, question or student_ans, style="analyze")
            st.write(corrected)

elif mode == "Flashcards":
    st.subheader("🗂 Auto-generated flashcards")
    pool = [r["text"] for r in random.sample(records, k=min(60,len(records)))]
    n = st.slider("How many flashcards", 3, 30, 8)
    if st.button("Generate flashcards"):
        cards = generate_flashcards(pool, n=n)
        if not cards:
            st.warning("Could not generate flashcards. Try increasing pool size.")
        else:
            for i,c in enumerate(cards, start=1):
                with st.expander(f"Card {i}: {c.get('q','(no q)')}"):
                    st.markdown(c.get("a",""))
            st.download_button("Download flashcards (JSON)", json.dumps(cards, indent=2), file_name="flashcards.json")

elif mode == "Quiz":
    st.subheader("📝 Generate MCQ quiz")
    pool = [r["text"] for r in random.sample(records, k=min(60,len(records)))]
    nq = st.slider("Number of questions", 3, 20, 5)
    if st.button("Generate quiz"):
        quiz = generate_quiz(pool, n_q=nq)
        st.text_area("Quiz (copy/paste)", quiz, height=420)
        st.download_button("Download quiz (TXT)", quiz, file_name="quiz.txt")

elif mode == "Summarize":
    st.subheader("🧾 Summarize a source file/chunks")
    sources = sorted({r["source"] for r in records})
    which = st.selectbox("Pick a source file", sources)
    howmany = st.slider("How many chunks to include", 1, 60, 10)
    target = [r["text"] for r in records if r["source"]==which][:howmany]
    if st.button("Generate summary"):
        ctx = "\n\n".join(target)
        prompt = f"Summarize key points for quick revision. Style: {style}"
        out = GENERATOR(prompt + "\n\nContext:\n" + ctx, max_length=400, do_sample=False)[0]["generated_text"]
        st.markdown(out)
        st.download_button("Download summary", out, file_name="summary.txt")

elif mode == "Concepts":
    st.subheader("🧠 Simple concept extraction (co-occurrence graph preview)")
    st.info("This is a light-weight extraction: it finds frequent multi-character words and shows co-occurrence. Not a full NER.")
    # Produce naive concept graph text (not interactive) for free approach
    texts = " ".join([r["text"] for r in records])
    words = [w.lower() for w in texts.split() if len(w)>5]
    freq = {}
    for w in words:
        w2 = ''.join(ch for ch in w if ch.isalnum())
        if len(w2)>4:
            freq[w2] = freq.get(w2,0)+1
    top = sorted(freq.items(), key=lambda x:-x[1])[:40]
    st.write("Top candidate concepts:", [t[0] for t in top])
    st.write("You can add a concept-graph visual later (requires pyvis/networkx).")

st.markdown("---")
st.caption("StudyMate (free local edition) — runs offline using Hugging Face models and local tools. Swap to larger models and GPU for faster, higher-quality responses.")
