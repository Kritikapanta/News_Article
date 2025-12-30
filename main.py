import requests
import numpy as np
import re
from collections import defaultdict
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
NEWS_API_KEY = "12ad376c3d5c4134b493484d2711eb14"
ARTICLES_PER_QUERY = 50
RESULTS_PER_PAGE = 10 

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    return re.findall(r"\b\w+\b", str(text).lower())

# =========================
# FETCH NEWS
# =========================
def fetch_news(query, page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    documents = {}
    metadata = {}
    idx = 0

    for article in data.get("articles", []):
        title = article.get("title") or ""
        desc = article.get("description") or ""
        content = title + " " + desc

        if not content.strip():
            continue

        documents[idx] = content
        metadata[idx] = {
            "title": title,
            "url": article.get("url"),
            "source": article.get("source", {}).get("name"),
            "publishedAt": article.get("publishedAt")
        }
        idx += 1

    return documents, metadata

# =========================
# TIME DECAY
# =========================
def time_decay(published_at, decay_rate=0.04):
    if not published_at:
        return 1.0

    published_time = datetime.fromisoformat(
        published_at.replace("Z", "+00:00")
    )
    now = datetime.now(timezone.utc)
    age_hours = (now - published_time).total_seconds() / 3600

    return np.exp(-decay_rate * age_hours)

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
        for term in terms:
            freq[term] += 1

        for term, f in freq.items():
            index[term].append((doc_id, f))

    avg_dl = total_terms / max(1, len(documents))
    return index, doc_lengths, avg_dl

# =========================
# BM25
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
# RM3 QUERY EXPANSION (SAFE)
# =========================
def expand_query_rm3(query, bm25_scores, documents, top_docs=10, top_terms=5):
    if len(bm25_scores) < 3:
        return query

    ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = min(top_docs, len(ranked))

    term_freq = defaultdict(int)
    for doc_id, _ in ranked[:top_docs]:
        for term in preprocess(documents[doc_id]):
            term_freq[term] += 1

    expansion_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:top_terms]
    expanded_query = query + " " + " ".join(t for t, _ in expansion_terms)

    return expanded_query

# =========================
# SEARCH PIPELINE
# =========================
def search_news(query):
    print("\nFetching news...")
    documents, metadata = fetch_news(query, ARTICLES_PER_QUERY)

    if not documents:
        print("No articles found.")
        return

    index, doc_lengths, avg_dl = build_index(documents)

    bm25_scores = score_BM25(query, index, doc_lengths, avg_dl)
    expanded_query = expand_query_rm3(query, bm25_scores, documents)
    rm3_scores = score_BM25(expanded_query, index, doc_lengths, avg_dl)

    # FINAL SCORING (ALL DOCS)
    final_scores = {}
    for doc_id in documents.keys():
        bm25_score = rm3_scores.get(doc_id, 0)
        decay = time_decay(metadata[doc_id]["publishedAt"])
        final_scores[doc_id] = 0.7 * bm25_score + 0.3 * decay

    ranked_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # =========================
    # PAGINATION (GOOGLE STYLE)
    # =========================
    total_results = len(ranked_docs)
    total_pages = (total_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE

    print(f"\nAbout {total_results} results found\n")

    page = 1
    while True:
        start = (page - 1) * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE
        page_docs = ranked_docs[start:end]

        if not page_docs:
            break

        print(f"--- PAGE {page}/{total_pages} ---\n")
        for rank, (doc_id, score) in enumerate(page_docs, start=start + 1):
            print(f"{rank}. {metadata[doc_id]['title']}")
            print(f"   Source: {metadata[doc_id]['source']}")
            print(f"   URL: {metadata[doc_id]['url']}\n")

        cmd = input("Press Enter for next page or type 'q' to quit: ")
        if cmd.lower() == "q":
            break

        page += 1

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    print("===== INTERNET NEWS SEARCH ENGINE =====")

    while True:
        query = input("\nEnter search query (or 'exit'): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        search_news(query)
