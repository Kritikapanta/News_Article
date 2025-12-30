import requests
import numpy as np
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
NEWS_API_KEY = "12ad376c3d5c4134b493484d2711eb14"
ARTICLES_PER_QUERY = 50

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# =========================
# FETCH NEWS FROM INTERNET
# =========================
def fetch_news(query, page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    documents = {}
    metadata = {}

    for i, article in enumerate(data.get("articles", [])):
        content = (article["title"] or "") + " " + (article["description"] or "")
        documents[i] = content
        metadata[i] = {
            "title": article["title"],
            "url": article["url"],
            "source": article["source"]["name"]
        }

    return documents, metadata

# =========================
# BUILD INVERTED INDEX
# =========================
def build_index(documents):
    index = defaultdict(list)
    doc_lengths = {}
    total_terms = 0

    for doc_id, text in documents.items():
        terms = preprocess(text)
        doc_lengths[doc_id] = len(terms)
        total_terms += len(terms)

        freq = defaultdict(int)
        for t in terms:
            freq[t] += 1

        for term, f in freq.items():
            index[term].append((doc_id, f))

    avg_dl = total_terms / len(documents)
    return index, doc_lengths, avg_dl

# =========================
# BM25 SCORING
# =========================
def score_BM25(query, index, doc_lengths, avg_dl, k1=1.5, b=0.75):
    scores = defaultdict(float)
    N = len(doc_lengths)
    query_terms = preprocess(query)

    for term in query_terms:
        if term not in index:
            continue

        df = len(index[term])
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

        for doc_id, f in index[term]:
            denom = f + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_dl))
            scores[doc_id] += idf * (f * (k1 + 1)) / denom

    return scores

# =========================
# RM3 QUERY EXPANSION
# =========================
def expand_query_rm3(query, bm25_scores, documents, top_docs=5, top_terms=5):
    top_documents = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:top_docs]

    term_freq = defaultdict(int)
    for doc_id, _ in top_documents:
        for term in preprocess(documents[doc_id]):
            term_freq[term] += 1

    expansion_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:top_terms]
    expanded_query = query + " " + " ".join([t for t, _ in expansion_terms])

    return expanded_query

# =========================
# SEARCH PIPELINE
# =========================
def search_news(query):
    print("\nFetching news from the internet...")
    documents, metadata = fetch_news(query, ARTICLES_PER_QUERY)

    if not documents:
        print("No articles found.")
        return

    index, doc_lengths, avg_dl = build_index(documents)

    # BM25
    bm25_scores = score_BM25(query, index, doc_lengths, avg_dl)

    # RM3
    expanded_query = expand_query_rm3(query, bm25_scores, documents)
    rm3_scores = score_BM25(expanded_query, index, doc_lengths, avg_dl)

    ranked_docs = sorted(rm3_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTOP NEWS RESULTS (BM25 + RM3)\n")
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        print(f"{rank}. {metadata[doc_id]['title']}")
        print(f"   Source: {metadata[doc_id]['source']}")
        print(f"   URL: {metadata[doc_id]['url']}\n")

# MAIN LOOP
if __name__ == "__main__":
    print("INTERNET NEWS SEARCH ENGINE")

    while True:
        user_query = input("\nEnter search query (or 'exit'): ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        search_news(user_query)