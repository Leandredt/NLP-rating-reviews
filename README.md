# NLP Project 2 — Insurance Review Analysis

End-to-end NLP pipeline on 35,000 French insurance customer reviews, from preprocessing to a fully interactive Streamlit application.

**ESILV A4 — NLP 2026**

---

## Overview

| | |
|---|---|
| **Dataset** | 35,000 French insurance reviews (translated to English) |
| **Task** | Star rating prediction (1–5), topic analysis, semantic search, summarization, RAG, QA |
| **Best model** | TF-IDF + Logistic Regression |
| **App** | Streamlit — prediction, explanation, summarization, IR, RAG, QA |

---

## Project Structure

```
├── project2.ipynb          # Main notebook (full pipeline)
├── app.py                  # Streamlit application
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── lr_model.pkl            # Saved Logistic Regression model
├── word2vec_insurance.model# Word2Vec trained on corpus
├── lda_model               # LDA topic model (7 topics)
└── fig_*.png               # Saved visualizations
```

---

## Pipeline

### 1. Data Cleaning & EDA
- Spell correction, stopword removal, lemmatization (spaCy)
- Translation: French → English
- Rating distribution, top N-grams, wordcloud

### 2. Topic Modeling (LDA)
- Gensim LDA with coherence-based topic selection → **7 topics**
- Topics: Claims Processing, Customer Service, Coverage, Enrollment, Cancellation, Pricing, Other
- Topic–rating heatmap shows clear thematic patterns per star rating

### 3. Embeddings
- Custom **Word2Vec** trained on 35k insurance reviews (100d)
- GloVe 100d (generic) for comparison
- t-SNE visualization of domain vocabulary clusters
- PyTorch models: trainable embedding layer + pre-trained W2V initialization

### 4. Supervised Models

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| TF-IDF + LR | ~0.50 | ~0.49 |
| Word2Vec + LR | ~0.44 | ~0.43 |
| TF-IDF + LDA + LR | ~0.50 | ~0.49 |
| PyTorch Embed Layer | ~0.45 | ~0.44 |
| PyTorch Pre-trained W2V | ~0.46 | ~0.45 |
| DistilBERT (500 samples) | ~0.52 | ~0.51 |

### 5. LLM Comparison

| Model | Task | Quality (1–5) | Speed |
|---|---|---|---|
| DistilBART-CNN-12-6 | Summarization | 4 | Medium |
| T5-small | Summarization | 3 | Fast |
| Flan-T5-small | RAG generation | 3 | Fast |
| Flan-T5-base | RAG generation | 4 | Medium |

### 6. Results Interpretation
- **Error analysis**: 3-star reviews most misclassified (ambiguous vocabulary); confusion concentrated between adjacent classes
- **Sentiment detection**: 1–2★ → Negative, 3★ → Neutral, 4–5★ → Positive
- **Classical models + themes**: LDA topics add marginal value over TF-IDF, strongest signal on 3-star reviews
- **Embedding analysis**: domain-trained W2V clusters insurance terms correctly (claim/refused/delay vs fast/efficient/recommend)

---

## Streamlit Application

### Tab 1 — Analysis & Summary
- **Prediction**: star rating + category (zero-shot BART) + LIME word-level explanation
- **Summarization**: DistilBART on any review text

### Tab 2 — Search & QA
- **Semantic Search (IR)**: cosine similarity on Word2Vec document vectors
- **RAG**: retrieve relevant reviews → Flan-T5 generates a grounded answer
- **Question Answering**: RoBERTa extracts answer span from retrieved context

### Run the app
```bash
# Install dependencies
pip install streamlit transformers gensim scikit-learn lime torch pandas

# Launch
cd Project2
streamlit run app.py
```

> **Note:** Models load on first run (~2–3 min). Keep the app open between queries.

---

## Requirements

```
pandas
numpy
scikit-learn
gensim
torch
transformers
streamlit
lime
spacy
nltk
seaborn
matplotlib
```

---

## Key Files Not Included

Due to size limits, the following are excluded from the repo:
- `reviews_cleaned.csv` (31 MB) — regenerate by running notebook cells 1–12
- `glove.6B.100d.txt` (330 MB) — download from [GloVe](https://nlp.stanford.edu/projects/glove/)
- `Traduction avis clients/` — original Excel source files
