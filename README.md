# Sales Playbook Assistant

A customer knowledge assistant: ask a question about a company, get an answer
pulled from that company's own documents, with the source cited. Built as a
portfolio project for AI solutions and forward-deployed engineering roles.

## What it is

This is the **retrieval layer of a RAG (Retrieval-Augmented Generation)
system**, built with classic ML (TF-IDF) so it runs anywhere with no API key
and no GPU:

1. **Ingest** — Markdown docs in `data/` (a fictional company, Northwind
   Supply Co.: pricing, onboarding, integrations, security, returns,
   troubleshooting, plus a sales playbook).
2. **Chunk** — one chunk per document section.
3. **Index** — TF-IDF vectors with light stemming (`costs` matches `cost`,
   `syncing` matches `sync`) and section headings weighted above body text.
4. **Retrieve** — a question is ranked against every chunk by cosine
   similarity; the top matches are returned with relevance scores.
5. **Answer (extractive)** — the best passage is quoted, with sources. If
   nothing matches well, it says so instead of making something up.

## What it is NOT (honest scope)

There is no LLM in this version. The generation step — where a model writes a
fluent answer from the retrieved passages — is a marked plug-in point in
`assistant.py` (`generate_answer()`). Swap the extractive return for an LLM
call (OpenAI, Anthropic, Bedrock, etc.) and the architecture is unchanged.
That swap is exactly the kind of customization a Forward Deployed Engineer
does on a customer's data.

## Run it

```bash
pip install -r requirements.txt

# Command line
python3 assistant.py "How much does the Pro plan cost?"

# Web demo
python3 app.py
# open http://127.0.0.1:5000
```

Try the questions in `demo_questions.txt`, or ask your own. Ask something
not in the docs (e.g. "What is the capital of France?") to see the
no-answer fallback — grounded answers only, no hallucinations.

## Project layout

```
assistant.py        core logic: chunking, TF-IDF index, search, answer
app.py              Flask web demo + JSON API (POST /api/ask)
templates/index.html  single-page UI
data/               the knowledge base (8 fictional company docs)
demo_questions.txt  questions to try
requirements.txt    flask, scikit-learn
```

## Next steps (if you want to go further)

1. Plug in an LLM for generative answers (needs an API key).
2. Swap TF-IDF for embeddings (e.g. sentence-transformers) for
   meaning-based rather than word-based matching.
3. Deploy it (Render, Railway, or AWS free tier) so there's a live link on
   your resume.
4. Put it on GitHub and add a Projects line to your resume.
