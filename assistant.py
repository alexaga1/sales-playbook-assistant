"""
Sales Playbook Assistant — retrieval layer of a RAG (Retrieval-Augmented
Generation) system.

What it does:
    1. Loads a company's knowledge base (Markdown docs in ./data).
    2. Splits each doc into chunks (one chunk per section).
    3. Indexes the chunks with TF-IDF vectors.
    4. For a question, finds the most relevant chunks by cosine similarity
       and returns the best passage plus its sources, with scores.

Honest scope note:
    This is the RETRIEVAL half of RAG, built with classic ML (TF-IDF) so it
    runs anywhere with no API key and no GPU. The GENERATION half — where an
    LLM writes a fluent answer from the retrieved passages — is a deliberate
    plug-in point: see generate_answer(). Swap the extractive return for an
    LLM call (OpenAI, Anthropic, Bedrock, etc.) and the architecture is
    unchanged. That swap is exactly the kind of customization a Forward
    Deployed Engineer does on a customer's data.

Run from the command line:
    python3 assistant.py "What is the return policy?"
Run the web demo:
    python3 app.py   ->  http://127.0.0.1:5000
"""
import os
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIN_SCORE = 0.10  # below this, the question doesn't match anything well
TOP_K = 3


def _stem(word):
    """Light stemmer so 'costs' matches 'cost' and 'syncing' matches 'sync'."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS]


def load_chunks(data_dir=DATA_DIR):
    """Split every .md file into one chunk per section (## heading)."""
    chunks = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # Split on level-2 headings; the text before the first heading is
        # only a chunk if it has body content beyond its title line.
        parts = re.split(r"\n## ", text)
        intro = parts[0].strip()
        intro_lines = intro.split("\n", 1)
        intro_body = intro_lines[1].strip() if len(intro_lines) > 1 else ""
        if intro_body:
            title = intro_lines[0].lstrip("# ").strip()
            chunks.append({"source": fname, "section": title or "Overview",
                           "text": intro_body})
        for part in parts[1:]:
            lines = part.strip().split("\n", 1)
            section = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if body:
                chunks.append({"source": fname, "section": section,
                               "text": body})
    return chunks


class KnowledgeAssistant:
    """TF-IDF retrieval over the knowledge base."""

    def __init__(self, data_dir=DATA_DIR):
        self.chunks = load_chunks(data_dir)
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize,
                                          token_pattern=None,
                                          ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(
            # Index the section heading together with the body, with the
            # heading repeated so a section title match (e.g. "Plan Prices"
            # for a pricing question) outweighs a passing mention elsewhere.
            [c["section"] + " " + c["section"] + " " + c["text"]
             for c in self.chunks])

    def search(self, question, top_k=TOP_K):
        """Return the top-k chunks ranked by cosine similarity."""
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i],
                        reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] >= MIN_SCORE:
                results.append({**self.chunks[i],
                                "score": round(float(scores[i]), 3)})
        return results

    def generate_answer(self, question):
        """
        EXTRACTIVE answer: quote the best retrieved passage.

        PLUG-IN POINT for generative AI: replace the return below with an
        LLM call whose prompt is:
            "Answer the question using ONLY the passages below. Cite the
             source document for each fact. If the passages don't contain
             the answer, say so."
            + the retrieved passages.
        Everything else — ingestion, chunking, retrieval, ranking, citation
        — stays exactly the same.
        """
        results = self.search(question)
        if not results:
            return {
                "answer": ("I couldn't find that in the knowledge base. "
                           "Try rephrasing, or ask about pricing, onboarding, "
                           "integrations, security, returns, or troubleshooting."),
                "sources": [],
            }
        best = results[0]
        return {
            "answer": best["text"],
            "sources": results,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 assistant.py "your question here"')
        sys.exit(1)
    assistant = KnowledgeAssistant()
    out = assistant.generate_answer(" ".join(sys.argv[1:]))
    print("\nANSWER:\n" + out["answer"])
    if out["sources"]:
        print("\nSOURCES:")
        for s in out["sources"]:
            print(f"  - {s['source']} > {s['section']} (relevance {s['score']})")
