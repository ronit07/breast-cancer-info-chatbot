"""Breast cancer information chatbot (retrieval-augmented, answers only from NCI pages).

Pipeline
  1. Load the notes in data/ and split them at section headings (one chunk per topic).
  2. Expand the question with medical synonyms (mammogram ~ mammography ~ screening).
  3. Rank chunks two ways, TF-IDF and BM25, and merge the rankings (reciprocal rank fusion).
  4. Safety layer: emergency wording gets an emergency message; personal symptoms get a
     "see your doctor" note; weak matches are declined instead of guessed.
  5. Answer with the best sentences from the top chunk(s) and cite the NCI page.
     Optional: if ANTHROPIC_API_KEY is set, Claude writes the answer from the same chunks.
"""
import os, re, glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIN_SCORE = 0.08
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as ENGLISH_STOP
DISCLAIMER = ("General information from National Cancer Institute pages. "
              "Not medical advice; please talk with your doctor about your situation.")

SYNONYMS = {
    "mammogram": "mammography screening x-ray", "mammography": "mammogram screening x-ray",
    "chemo": "chemotherapy", "tnbc": "triple negative", "mri": "magnetic resonance imaging",
    "gene": "BRCA genetic inherited", "genetic": "BRCA gene inherited", "hereditary": "inherited BRCA",
    "lump": "lump breast change", "dense": "density", "risk": "chance probability lifetime",
    "chance": "risk probability", "odds": "risk probability", "tamoxifen": "hormone therapy risk-reducing",
    "biopsy": "biopsy suspicious BI-RADS", "result": "report BI-RADS", "results": "report BI-RADS",
    "harm": "harms false-positive overdiagnosis", "downside": "harms false-positive overdiagnosis",
    "cancerous": "benign noncancerous cancer", "noncancerous": "benign",
}
EMERGENCY = re.compile(r"\b(chest pain|can'?t breathe|trouble breathing|suicid|want to die|overdose|heavy bleeding)\b", re.I)
PERSONAL = re.compile(r"\b(i (found|feel|have|noticed)|my (breast|nipple|mammogram|result|doctor)|lump in my|am i at risk|do i have)\b", re.I)
FOLLOWUP = re.compile(r"^(and|what about|how about|what if|why|then|is it|does it|can it)\b|\b(it|that|this|they|those)\b", re.I)


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def expand(q):
    extra = [SYNONYMS[w] for w in tokens(q) if w in SYNONYMS]
    if re.search(r"\b(at|age|aged|turn(ed|ing)?|over|under)\s*(age\s*)?\d{2}\b|\b\d{2}s?\s*(years|yrs|year old)", q, re.I):
        extra.append("age ages starting recommended every years")
    return q + " " + q + " " + " ".join(extra)


def load_chunks(data_dir=DATA_DIR):
    chunks = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        text = open(path, encoding="utf-8").read()
        title = re.search(r"^SOURCE:\s*(.+)$", text, re.M).group(1).strip()
        url = re.search(r"^URL:\s*(.+)$", text, re.M).group(1).strip()
        for part in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, _, body = part.partition("\n")
            chunks.append({"title": title, "url": url, "heading": heading.strip(),
                           "text": body.strip(), "file": os.path.basename(path)})
    return chunks


class Bot:
    def __init__(self, chunks=None):
        self.chunks = chunks or load_chunks()
        docs = [f"{c['heading']}. {c['heading']}. {c['text']}" for c in self.chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                   token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vec.fit_transform(docs)
        self.bm25 = BM25Okapi([tokens(d) for d in docs])

    def retrieve(self, question, k=3):
        q = expand(question)
        tfidf = (self.matrix @ self.vec.transform([q]).T).toarray().ravel()
        bm = self.bm25.get_scores(tokens(q))
        fused = np.zeros(len(self.chunks))
        for scores in (tfidf, bm):
            for rank, i in enumerate(np.argsort(-scores)):
                fused[i] += 1.0 / (60 + rank)
        top = np.argsort(-fused)[:k]
        self.best_tfidf = float(tfidf.max())
        return [(self.chunks[i], float(tfidf[i])) for i in top]

    def _pick(self, question, hits, n=3):
        """Score every sentence in the top chunks; keep the best few, in reading order."""
        qv = self.vec.transform([question + " " + question])  # original wording: rare words matter most
        cand = []
        hits = [h for h in hits if h[1] >= 0.6 * hits[0][1]]  # ignore weakly related chunks
        for rank, (chunk, _) in enumerate(hits):
            for pos, s in enumerate(re.split(r"(?<=[.!?])\s+", chunk["text"])):
                if len(s) > 15:
                    sc = float((self.vec.transform([s]) @ qv.T).toarray()[0, 0])
                    cand.append((sc - 0.01 * rank, rank, pos, s, chunk))
        best = sorted([c for c in cand if c[0] > 0], key=lambda x: -x[0])[:n]
        return [(s, chunk) for _, _, _, s, chunk in sorted(best, key=lambda x: (x[1], x[2]))]

    def _extractive(self, question, hits):
        return " ".join(s for s, _ in self._pick(question, hits))

    def _llm(self, question, hits):
        import anthropic
        ctx = "\n\n".join(f"[{i+1}] {c['title']} - {c['heading']}\n{c['text']}" for i, (c, _) in enumerate(hits))
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("CHATBOT_MODEL", "claude-sonnet-5-5"), max_tokens=500,
            system="Answer ONLY from the numbered sources and cite them like [1]. If they do not "
                   "answer the question, say so. Never give personal medical advice.",
            messages=[{"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"}])
        return msg.content[0].text

    def ask(self, question, history=None, k=3):
        question = question.strip()
        if EMERGENCY.search(question):
            return {"answer": "If this is an emergency, call 911 (or your local emergency number) now. "
                              "I can't help with emergencies.", "sources": [], "mode": "emergency", "score": 0.0}
        search_q = question
        if history and len(tokens(question)) <= 7 and FOLLOWUP.search(question):
            search_q = history[-1] + " " + question  # follow-up: reuse the last question's topic
        hits = self.retrieve(search_q, k)
        known = {w for w in tokens(" ".join(c["text"] + " " + c["heading"] for c in self.chunks))}
        content = [w for w in tokens(search_q) if w not in ENGLISH_STOP and len(w) > 2]
        if self.best_tfidf < MIN_SCORE or not any(w in known for w in content):
            return {"answer": "I couldn't find that in the NCI pages I have. I only cover breast "
                              "cancer screening, mammograms, risk, BRCA genes, treatment types, receptor "
                              "testing and breast changes. For anything else, please ask a doctor.",
                    "sources": [], "mode": "declined", "score": hits[0][1]}
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                answer, mode = self._llm(search_q, hits), "llm"
            else:
                answer, mode = self._extractive(search_q, hits), "extractive"
        except Exception as e:
            answer, mode = self._extractive(search_q, hits), "extractive"
        if PERSONAL.search(question):
            answer = ("I can't give personal medical advice. NCI says to contact your doctor if you notice "
                      "any changes in your breasts. Here is the general information: ") + answer
        seen, sources = set(), []
        used = [c for _, c in self._pick(search_q, hits)] or [c for c, _ in hits[:1]]
        for c in used:
            if c["url"] not in seen:
                seen.add(c["url"]); sources.append({"title": f"{c['title']}: {c['heading']}", "url": c["url"]})
        return {"answer": answer, "sources": sources, "mode": mode, "score": hits[0][1]}


def main():
    bot, history = Bot(), []
    print("Breast Cancer Info Bot\n" + DISCLAIMER + "\nType 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if q.lower() in {"quit", "exit", ""}:
            break
        r = bot.ask(q, history); history.append(q)
        print(f"\nBot: {r['answer']}")
        for s in r["sources"]:
            print(f"  Source: {s['title']} <{s['url']}>")
        print()


if __name__ == "__main__":
    main()
