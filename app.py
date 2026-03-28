"""
Project 2 — Insurance Reviews NLP App
Streamlit application: predict star rating + category + LIME explanation
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import pickle
import numpy as np
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Review Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_clf_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    return tfidf, lr

@st.cache_resource
def load_zero_shot():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1  # CPU
    )

@st.cache_resource
def load_summarization_model():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1 # CPU
    )

@st.cache_resource
def load_rag_generator():
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small", # A good small model for instruction following
        device=-1 # CPU
    )

@st.cache_resource
def load_qa_model():
    from transformers import pipeline
    return pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

@st.cache_resource
def load_cleaned_reviews():
    import pandas as pd
    return pd.read_csv("reviews_cleaned.csv")

@st.cache_resource
def load_word2vec_model():
    from gensim.models import Word2Vec
    return Word2Vec.load("word2vec_insurance.model")

@st.cache_resource
def get_doc_vectors(df, _w2v_model):
    def get_vec(text, model):
        if not isinstance(text, str):
            return np.zeros(model.vector_size)
        tokens = text.split()
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if not vecs:
            return np.zeros(model.vector_size)
        return np.mean(vecs, axis=0).astype(float)

    # Keep only rows with valid English text and a numeric note
    df_valid = df[
        df['text_clean'].notna() &
        df['avis_en'].notna() &
        df['note'].notna()
    ].reset_index(drop=True)
    doc_vectors = np.array(
        [get_vec(str(t), _w2v_model) for t in df_valid['text_clean']],
        dtype=float
    )
    return doc_vectors, df_valid

# ── Text cleaning (must match notebook preprocessing) ────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── LIME explanation ─────────────────────────────────────────────────────────
def get_lime_html(text: str, tfidf, lr, pred_label: int) -> str:
    from lime.lime_text import LimeTextExplainer
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(tfidf, lr)
    class_names = [str(i) for i in range(1, 6)]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text, pipe.predict_proba,
        num_features=15, labels=[pred_label - 1]
    )
    words_weights = exp.as_list(label=pred_label - 1)
    return words_weights, exp.as_html()

# ── Semantic Search ──────────────────────────────────────────────────────────
def _vec_from_text(text, model):
    if not isinstance(text, str):
        return np.zeros(model.vector_size, dtype=float)
    tokens = text.split()
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vecs, axis=0).astype(float)


_EN_STOPWORDS = {
    "the", "is", "are", "was", "were", "i", "my", "me", "we", "have", "has",
    "this", "that", "a", "an", "and", "or", "not", "it", "they", "their",
    "you", "your", "he", "she", "with", "for", "of", "to", "in", "on", "at",
    "be", "been", "do", "did", "no", "but", "so", "very", "more", "than",
    "after", "all", "which", "who", "when", "what", "how", "there", "from",
}

def _is_english(text: str) -> bool:
    """Detect English by presence of common English stopwords."""
    if not isinstance(text, str) or len(text) < 10:
        return False
    words = set(text.lower().split())
    matches = words & _EN_STOPWORDS
    return len(matches) >= 2


def semantic_search(query: str, df_valid, doc_vectors, w2v_model, top_k: int = 5):
    from sklearn.metrics.pairwise import cosine_similarity

    q_clean = clean_text(query)
    q_vec = _vec_from_text(q_clean, w2v_model).reshape(1, -1)

    sims = cosine_similarity(q_vec, doc_vectors)[0]

    # Take top candidates (extra buffer to allow filtering non-English)
    candidate_idx = sims.argsort()[-(top_k * 4):][::-1]

    results = []
    for idx in candidate_idx:
        avis_en = df_valid.loc[idx, 'avis_en']
        if not _is_english(str(avis_en)):
            continue
        results.append({
            'score': float(sims[idx]),
            'note': df_valid.loc[idx, 'note'],
            'avis': df_valid.loc[idx, 'avis'],
            'avis_en': avis_en,
        })
        if len(results) >= top_k:
            break
    return results

# ── Star display ─────────────────────────────────────────────────────────────
STAR_COLORS = {1: "#d73027", 2: "#f46d43", 3: "#fdae61", 4: "#74add1", 5: "#313695"}
STAR_LABELS = {1: "Very Poor", 2: "Poor", 3: "Average", 4: "Good", 5: "Excellent"}

def safe_note(note) -> int:
    try:
        return int(float(note))
    except (ValueError, TypeError):
        return 0

def render_stars(note) -> str:
    n = safe_note(note)
    if n == 0:
        return '<span style="font-size:2.5rem; color:#999">☆☆☆☆☆</span>'
    filled = "★" * n
    empty = "☆" * (5 - n)
    color = STAR_COLORS.get(n, "#999")
    return f'<span style="font-size:2.5rem; color:{color}">{filled}{empty}</span>'

# ── Categories ───────────────────────────────────────────────────────────────
CATEGORIES = [
    "Pricing and premiums",
    "Claims handling",
    "Customer service",
    "Contract cancellation",
    "Other"
]

# ── App layout ───────────────────────────────────────────────────────────────
st.title("🛡️ Insurance Review Analyzer")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Models used:**\n"
        "- Star prediction: TF-IDF + Logistic Regression\n"
        "- Category: zero-shot (facebook/bart-large-mnli)\n"
        "- Explainability: LIME\n"
        "- Summarization: DistilBART\n"
        "- Semantic search: Word2Vec\n"
        "- RAG: Flan-T5\n"
        "- QA: RoBERTa\n\n"
        "**Dataset:** 34,999 French insurance reviews (translated to English)"
    )
    st.divider()
    st.markdown("*Project 2 — ESILV A4 NLP 2026*")

# Check models exist
models_ready = os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("lr_model.pkl")
if not models_ready:
    st.error(
        "⚠️ Models not found. Please run `project2.ipynb` Phase 4 first "
        "to generate `tfidf_vectorizer.pkl` and `lr_model.pkl`."
    )
    st.stop()

tfidf, lr = load_clf_models()

# Load reviews once for dataset examples
df_reviews = load_cleaned_reviews()
w2v_model_ir = load_word2vec_model()
doc_vectors, df_valid = get_doc_vectors(df_reviews, w2v_model_ir)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analysis & Summary", "🔎 Search & QA"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction + Summary
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "Enter a customer review (in English) to get a predicted **star rating**, "
        "**category detection**, and word-level **LIME explanation**."
    )

    if "example_text" not in st.session_state:
        st.session_state.example_text = ""
    if "example_real_star" not in st.session_state:
        st.session_state.example_real_star = None

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("**Try these examples from the dataset:**")
        st.caption("(real rating hidden — can you predict it?)")
        # Pick one review per star rating from the dataset
        dataset_examples = []
        for star in [1, 2, 3, 4, 5]:
            subset = df_reviews[df_reviews['note'] == star]['avis_en'].dropna()
            if not subset.empty:
                sample = subset.iloc[0]
                if isinstance(sample, str) and len(sample) > 20:
                    dataset_examples.append((star, sample))

        for star, text in dataset_examples:
            preview = text[:65] + "…" if len(text) > 65 else text
            if st.button(preview, key=f"ds_ex_{star}"):
                st.session_state["example_text"] = text
                st.session_state["example_real_star"] = star
                st.rerun()

    with col1:
        review_text = st.text_area(
            "📝 Enter your insurance review:",
            height=160,
            value=st.session_state.example_text,
            placeholder="e.g. I filed a claim three months ago and still haven't received any response. "
                         "Terrible customer service, will never renew.",
        )

    predict_btn = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

    if predict_btn and review_text.strip():
        cleaned = clean_text(review_text)

        with st.spinner("Analyzing..."):
            X_vec = tfidf.transform([cleaned])
            pred_note = int(lr.predict(X_vec)[0])
            proba = lr.predict_proba(X_vec)[0]

            try:
                zs_clf = load_zero_shot()
                zs_result = zs_clf(review_text[:512], candidate_labels=CATEGORIES)
                top_category = zs_result["labels"][0]
                top_cat_score = zs_result["scores"][0]
                cat_scores = dict(zip(zs_result["labels"], zs_result["scores"]))
            except Exception as e:
                top_category = "Unknown"
                top_cat_score = 0.0
                cat_scores = {}
                st.warning(f"Zero-shot model unavailable: {e}")

            try:
                words_weights, lime_html = get_lime_html(cleaned, tfidf, lr, pred_note)
                lime_ok = True
            except Exception as e:
                lime_ok = False
                lime_error = str(e)

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown("### ⭐ Predicted Rating")
            st.markdown(render_stars(pred_note), unsafe_allow_html=True)
            color = STAR_COLORS.get(pred_note, "#999")
            st.markdown(
                f'<p style="color:{color}; font-size:1.2rem; font-weight:bold;">'
                f'★{pred_note}/5 — {STAR_LABELS[pred_note]}</p>',
                unsafe_allow_html=True
            )
            real_star = st.session_state.get("example_real_star")
            if real_star is not None:
                real_color = STAR_COLORS.get(real_star, "#999")
                match = "✅" if real_star == pred_note else "❌"
                st.markdown(
                    f'<p style="color:{real_color}; font-size:0.95rem;">'
                    f'{match} Real rating: ★{real_star}/5 — {STAR_LABELS[real_star]}</p>',
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown("### 📊 Rating Probabilities")
            for i, p in enumerate(proba, start=1):
                c = STAR_COLORS.get(i, "#999")
                st.markdown(
                    f'<div style="margin:3px 0">'
                    f'<span style="color:{c}">★{i}</span> '
                    f'<div style="display:inline-block; background:{c}; '
                    f'width:{p*200:.0f}px; height:14px; vertical-align:middle; border-radius:3px"></div>'
                    f' {p:.1%}</div>',
                    unsafe_allow_html=True
                )

        with res_col3:
            st.markdown("### 🏷️ Detected Category")
            st.markdown(f"**{top_category}**")
            if cat_scores:
                for cat, score in sorted(cat_scores.items(), key=lambda x: -x[1]):
                    st.progress(float(score), text=f"{cat}: {score:.1%}")

        st.divider()
        st.markdown("### 🔬 Word-level Explanation (LIME)")

        if lime_ok:
            pos_words = {w: s for w, s in words_weights if s > 0}
            neg_words = {w: s for w, s in words_weights if s < 0}

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("**🟢 Words supporting this rating:**")
                for w, s in sorted(pos_words.items(), key=lambda x: -x[1])[:8]:
                    st.markdown(f"- `{w}` (+{s:.4f})")
            with col_neg:
                st.markdown("**🔴 Words opposing this rating:**")
                for w, s in sorted(neg_words.items(), key=lambda x: x[1])[:8]:
                    st.markdown(f"- `{w}` ({s:.4f})")

            with st.expander("📄 Full LIME explanation (interactive HTML)"):
                padded_html = lime_html.replace(
                    "<body>",
                    "<body><style>body{margin:0;padding:0 0 0 120px;} "
                    ".lime{overflow-x:visible!important;} "
                    "svg{overflow:visible!important;}</style>"
                )
                st.components.v1.html(padded_html, height=500, scrolling=True)
        else:
            st.warning(f"LIME explanation unavailable: {lime_error}")

    elif predict_btn:
        st.warning("Please enter a review text before analyzing.")

    # ── Summarization ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📝 Text Summarization")

    if "summary_example_text" not in st.session_state:
        st.session_state.summary_example_text = ""

    # Dataset examples with real stars — only long reviews worth summarizing
    st.markdown("**Try these examples from the dataset:**")
    for i, star in enumerate([1, 2, 3, 4, 5]):
        subset = df_reviews[df_reviews['note'] == star]['avis_en'].dropna()
        long_reviews = subset[subset.str.len() > 400]
        if not long_reviews.empty:
            sample = long_reviews.iloc[min(i * 5, len(long_reviews) - 1)]
            label = "★" * star + "☆" * (5 - star)
            preview = sample[:90] + "…"
            if st.button(f"{label}  {preview}", key=f"sum_ex_{star}"):
                st.session_state["summary_input"] = sample
                st.rerun()

    summary_col1, summary_col2 = st.columns([2, 1])

    with summary_col1:
        summary_text_input = st.text_area(
            "Enter text to summarize:",
            height=200,
            key="summary_input",
            placeholder="e.g. Copy-paste a long review here for a quick summary."
        )
    with summary_col2:
        min_length = st.slider("Minimum summary length (words)", 20, 80, 30)
        max_length = st.slider("Maximum summary length (words)", 60, 300, 120)
        summary_btn = st.button("✨ Generate Summary", type="secondary", use_container_width=True)

    if summary_btn and summary_text_input.strip():
        if len(summary_text_input.strip()) < 100:
            st.warning("Text is too short to summarize meaningfully. Please use a longer review.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summarizer = load_summarization_model()
                    # DistilBART max input is ~1024 tokens ≈ 3500 chars; truncate cleanly
                    input_text = summary_text_input.strip()[:3500]
                    summary_result = summarizer(
                        input_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )
                    st.success("Summary generated!")
                    st.info(summary_result[0]['summary_text'])
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    elif summary_btn:
        st.warning("Please enter text to summarize.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Search & QA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("Search through **34,999 insurance reviews** using semantic similarity, get AI-generated answers, or ask direct questions.")

    # ── Information Retrieval ──────────────────────────────────────────────
    st.markdown("### 🔎 Semantic Search (Information Retrieval)")

    ir_query = st.text_input(
        "Enter a query to find similar reviews:",
        placeholder="e.g. bad customer service claim"
    )
    ir_btn = st.button("🔍 Search Reviews", type="primary", use_container_width=True, key="ir_button")

    if ir_btn and ir_query.strip():
        with st.spinner("Searching for relevant reviews..."):
            try:
                search_results = semantic_search(ir_query, df_valid, doc_vectors, w2v_model_ir, top_k=5)
                if search_results:
                    st.subheader("Top 5 Similar Reviews:")
                    for i, res in enumerate(search_results):
                        st.markdown(f"**{i+1}. Score: {res['score']:.3f} | Rating: {render_stars(res['note'])}**", unsafe_allow_html=True)
                        with st.expander(f"Review {i+1} (★{res['note']})"):
                            st.write(f"**Original (FR):** {res['avis']}")
                            st.write(f"**English Translation:** {res['avis_en']}")
                        st.markdown("---")
                else:
                    st.info("No similar reviews found.")
            except Exception as e:
                st.error(f"Error during semantic search: {e}")
    elif ir_btn:
        st.warning("Please enter a query to search for reviews.")

    # ── RAG ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 Retrieval Augmented Generation (RAG)")

    rag_query = st.text_input(
        "Ask a question about the insurance reviews:",
        placeholder="e.g. What are the common complaints about claims handling?"
    )
    rag_btn = st.button("💬 Get Answer (RAG)", type="primary", use_container_width=True, key="rag_button")

    if rag_btn and rag_query.strip():
        with st.spinner("Retrieving information and generating answer..."):
            try:
                retrieved_reviews = semantic_search(rag_query, df_valid, doc_vectors, w2v_model_ir, top_k=3)

                if not retrieved_reviews:
                    st.info("No relevant reviews found to answer your question.")
                else:
                    context = ""
                    for i, review in enumerate(retrieved_reviews):
                        context += f"Review {i+1} (Rating: {review['note']}): {review['avis_en']}\n"

                    prompt = f"Given the following insurance reviews, answer the question: {rag_query}\n\nReviews:\n{context}\nAnswer:"

                    generator = load_rag_generator()
                    rag_answer = generator(prompt, max_length=150, min_length=30, do_sample=True)[0]['generated_text']

                    st.subheader("Generated Answer:")
                    st.write(rag_answer)

                    st.subheader("Sources:")
                    for i, review in enumerate(retrieved_reviews):
                        st.markdown(f"- **Review {i+1} (★{review['note']}):** {review['avis_en'][:150]}...")
                        with st.expander(f"Full Review {i+1} (★{review['note']})"):
                            st.write(f"**Original (FR):** {review['avis']}")
                            st.write(f"**English Translation:** {review['avis_en']}")

            except Exception as e:
                st.error(f"Error during RAG process: {e}")
    elif rag_btn:
        st.warning("Please enter a question for RAG.")

    # ── Question Answering ────────────────────────────────────────────────
    st.divider()
    st.markdown("### ❓ Question Answering")

    qa_question = st.text_input(
        "Ask a question to find an answer in the reviews:",
        placeholder="e.g. What is the main issue with customer service?"
    )
    qa_btn = st.button("🗣️ Get Answer (QA)", type="primary", use_container_width=True, key="qa_button")

    if qa_btn and qa_question.strip():
        with st.spinner("Retrieving context and finding answer..."):
            try:
                retrieved_reviews_qa = semantic_search(qa_question, df_valid, doc_vectors, w2v_model_ir, top_k=3)

                if not retrieved_reviews_qa:
                    st.info("No relevant reviews found to serve as context for your question.")
                else:
                    context_qa = ""
                    for i, review in enumerate(retrieved_reviews_qa):
                        context_qa += f"Review {i+1} (Rating: {review['note']}): {review['avis_en']}\n"

                    qa_pipeline = load_qa_model()
                    qa_result = qa_pipeline(question=qa_question, context=context_qa)

                    st.subheader("Answer:")
                    st.info(qa_result['answer'])
                    st.write(f"*(Confidence: {qa_result['score']:.2f})*")

                    st.subheader("Context Used:")
                    st.markdown(f"```\n{context_qa}\n```")

                    st.subheader("Source Reviews:")
                    for i, review in enumerate(retrieved_reviews_qa):
                        st.markdown(f"- **Review {i+1} (★{review['note']}):** {review['avis_en'][:150]}...")
                        with st.expander(f"Full Review {i+1} (★{review['note']})"):
                            st.write(f"**Original (FR):** {review['avis']}")
                            st.write(f"**English Translation:** {review['avis_en']}")

            except Exception as e:
                st.error(f"Error during QA process: {e}")
    elif qa_btn:
        st.warning("Please enter a question for QA.")

# Footer
st.divider()
st.caption("Project 2 — Insurance Reviews NLP Pipeline | ESILV A4 2026")
