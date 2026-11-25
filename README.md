# Breast Cancer Info Chatbot

A chatbot that answers questions about breast cancer using seven National Cancer Institute (cancer.gov)
pages, shows which page each answer came from, and says so when it cannot answer.
Built Nov 2026 as a learning project. It gives general information, not medical advice.

## Run it
    pip install -r requirements.txt
    python server.py      # web interface at http://localhost:8000
    python chatbot.py     # or chat in the terminal
    python eval.py        # run the test set

## How it works
1. **Documents** (`data/`): condensed notes from seven NCI pages: treatment overview, screening, mammograms
   and BI-RADS results, risk statistics, BRCA genes, receptor testing (ER/PR/HER2), and common breast changes.
   Each file keeps its source link.
2. **Chunking**: files are split at section headings, so one chunk is one topic.
3. **Search**: the question is expanded with medical synonyms (for example "chemo" and "chemotherapy"), then
   chunks are ranked with TF-IDF and BM25 and the two rankings are merged.
4. **Answer**: the best-matching sentences from the top chunk are returned with a link to the NCI page.
   If `ANTHROPIC_API_KEY` is set, Claude writes the answer from the same chunks instead (not tested here).
5. **Safety**: emergency wording gets an emergency message; personal questions (for example "I found a lump")
   get a note to contact a doctor; questions the pages do not cover are declined, not guessed.
6. **Follow-ups**: short follow-up questions ("What about at 50?") reuse the previous question's topic.

## Test results (`eval.py`, 30 questions)
- Right NCI page in the top 3 results: 30/30
- Key fact present in the answer: 25/30
- Follow-up questions routed correctly: 3/3; off-topic questions declined: 5/5
- Emergency and personal-question handling: 1/1 each

The questions were written by the builder from the same pages, so these numbers are a sanity check, not
proof the bot is reliable on real users' questions.

## Limitations
- Seven pages only, kept as condensed notes (paraphrased, not verbatim).
- Search matches words, not meaning, so answers to personal or oddly worded questions can be off, and
  extractive answers can read choppily. Embedding search and an LLM answer step are the next upgrades.
- Not a diagnostic tool. Always talk with a doctor.
