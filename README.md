# Breast Cancer Info Chatbot

A chatbot that answers questions about breast cancer using seven National Cancer Institute (cancer.gov)
pages, shows which page each answer came from, and says so when it cannot answer.
Live demo: https://breast-cancer-info-chatbot.vercel.app

Built Nov 2026 as a learning project. It gives general information, not medical advice.

## Run it
    pip install -r requirements.txt
    python server.py      # web interface at http://localhost:8000
    python chatbot.py     # or chat in the terminal
    python eval.py        # run the test set

## Deploy
The repo includes `api/index.py` and `vercel.json`, so it deploys to Vercel as-is (import the GitHub repo,
no settings needed). Search uses only `numpy` and `rank_bm25`, so it stays small.

## How it works
1. **Documents** (`data/`): condensed notes from seven NCI pages: treatment overview, screening, mammograms
   and BI-RADS results, risk statistics, BRCA genes, receptor testing (ER/PR/HER2), and common breast changes.
   Each file keeps its source link.
2. **Chunking**: files are split at section headings, so one chunk is one topic.
3. **Search**: words are lightly stemmed and the question is expanded with medical synonyms (for example
   "chemo" and "chemotherapy"). Chunks are ranked with TF-IDF and BM25 and the two rankings are merged.
4. **Answer**: the best-matching section is shown as written (never mixing text from different sections), with a
   link to the NCI page. Very long sections are trimmed to their most relevant sentences.
   Optional: `pip install anthropic` and set `ANTHROPIC_API_KEY`, and Claude writes the answer from the same
   section instead (not tested here).
5. **Safety**: emergency wording gets an emergency message; personal questions (for example "I found a lump")
   get a note to contact a doctor; questions the pages do not cover are declined, not guessed.
6. **Follow-ups**: short follow-up questions ("What about at 50?") reuse the previous question's topic.

## Test results (`eval.py`, 32 questions)
- Right NCI page in the top 3 results: 32/32
- Key fact present in the answer: 31/32
- Follow-up questions routed correctly: 3/3; off-topic questions declined: 5/5
- Emergency and personal-question handling: 1/1 each

The questions were written by the builder from the same pages, so these numbers are a sanity check, not
proof the bot is reliable on real users' questions.

## Limitations
- Seven pages only, kept as condensed notes (paraphrased, not verbatim).
- Answers are whole sections of the notes, so they can be longer than needed. Search matches words, not meaning, so answers to personal or oddly worded questions can be off, and
  extractive answers can read choppily. Embedding search and an LLM answer step are the next upgrades.
- Not a diagnostic tool. Always talk with a doctor.
