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
import os, re, glob, math
from collections import Counter
from rank_bm25 import BM25Okapi

# Common English words ignored by search (same list scikit-learn uses, copied so the app stays small).
STOP_WORDS = frozenset("""a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can cannot cant co con could couldnt cry de describe detail do done down due during each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen fifty fill find fire first five for former formerly forty found four from front full further get give go had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie if in inc indeed interest into is it its itself keep last latter latterly least less ltd made many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off often on once one only onto or other others otherwise our ours ourselves out over own part per perhaps please put rather re same see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under until up upon us very via was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you your yours yourself yourselves""".split())
ENGLISH_STOP = STOP_WORDS
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIN_SCORE = 0.08
DISCLAIMER = ("General information from National Cancer Institute pages. "
              "Not medical advice; please talk with your doctor about your situation.")

SYNONYMS = {
    "mammogram": "mammography", "mammography": "mammogram",
    "chemo": "chemotherapy", "tnbc": "triple negative", "mri": "magnetic resonance imaging",
    "gene": "BRCA genetic inherited", "genetic": "BRCA gene inherited", "hereditary": "inherited BRCA",
    "lump": "lump breast change", "dense": "density", "risk": "chance probability lifetime",
    "chance": "risk probability", "odds": "risk probability", "tamoxifen": "hormone therapy risk-reducing",
    "biopsy": "biopsy suspicious BI-RADS", "result": "report BI-RADS", "results": "report BI-RADS",
    "harm": "harms false-positive overdiagnosis", "downside": "harms false-positive overdiagnosis",
    "cancerous": "benign noncancerous cancer", "noncancerous": "benign",
    "triple": "ER PR HER2 negative subtype receptor", "child": "children under 18", "children": "under 18 child",
    "kid": "children under 18", "often": "every years", "frequently": "every years", "start": "age ages starting",
    "painful": "pain hurt compression", "hurt": "pain compression", "cost": "insurance", "radiation": "radiation dose x-ray",
}
EMERGENCY = re.compile(r"\b(chest pain|can'?t breathe|trouble breathing|suicid|want to die|overdose|heavy bleeding)\b", re.I)
PERSONAL = re.compile(r"\b(i (found|feel|have|noticed)|my (breast|nipple|mammogram|result|doctor)|lump in my|am i at risk|do i have)\b", re.I)
FOLLOWUP = re.compile(r"^(and|what about|how about|what if|why|then|is it|does it|can it)\b|\b(it|that|this|they|those)\b", re.I)


def stem(w):
    for suf in ("ing", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP]


def stem_text(text):
    return " ".join(tokens(text))


def expand(q):
    extra = [SYNONYMS[w] for w in re.findall(r"[a-z0-9]+", q.lower()) if w in SYNONYMS]
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


class TfIdf:
    """Small TF-IDF index (unigrams + bigrams, log-scaled term counts, cosine similarity)."""

    def __init__(self, docs):
        grams = [self._grams(d) for d in docs]
        df = Counter(g for gs in grams for g in set(gs))
        n = len(docs)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.vectors = [self._weigh(gs) for gs in grams]

    @staticmethod
    def _grams(text):
        toks = tokens(text)
        return toks + [a + " " + b for a, b in zip(toks, toks[1:])]

    def _weigh(self, grams):
        w = {g: (1 + math.log(c)) * self.idf[g] for g, c in Counter(grams).items() if g in self.idf}
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {g: v / norm for g, v in w.items()}

    def scores(self, text):
        q = self._weigh(self._grams(text))
        return [sum(v * d.get(g, 0.0) for g, v in q.items()) for d in self.vectors]


class Bot:
    def __init__(self, chunks=None):
        self.chunks = chunks or load_chunks()
        docs = [f"{c['heading']}. {c['heading']}. {c['text']}" for c in self.chunks]
        self.tfidf = TfIdf(docs)
        self.bm25 = BM25Okapi([tokens(d) for d in docs])
        self.sents = [(s, c, pos) for c in self.chunks
                      for pos, s in enumerate(re.split(r"(?<=[.!?])\s+", c["text"])) if len(s) > 15]
        self.sent_bm25 = BM25Okapi([tokens(s) for s, _, _ in self.sents])

    def retrieve(self, question, k=3):
        q = expand(question)
        tfidf = self.tfidf.scores(q)
        bm = list(self.bm25.get_scores(tokens(q)))
        fused = [0.0] * len(self.chunks)
        for scores in (tfidf, bm):
            for rank, i in enumerate(sorted(range(len(scores)), key=lambda j: -scores[j])):
                fused[i] += 1.0 / (60 + rank)
        if re.search(r"\bhow (often|frequent\w*)\b|\bevery how\b", question, re.I):  # schedule questions
            for i, c in enumerate(self.chunks):
                if re.search(r"\bevery\b.*\byears?\b", c["text"]):
                    fused[i] += 0.03
        top = sorted(range(len(fused)), key=lambda j: -fused[j])[:k]
        self.best_tfidf = float(max(tfidf))
        return [(self.chunks[i], float(tfidf[i])) for i in top]

    def _pick(self, question, hits, n=3):
        """BM25 over single sentences (rare words count most). Sentences from the top-ranked
        chunks get a small boost; the best few are shown in reading order."""
        scores = self.sent_bm25.get_scores(tokens(expand(question)))
        top_ids = {id(c): r for r, (c, _) in enumerate(hits[:3])}
        cand = []
        for i, (s, ch, pos) in enumerate(self.sents):
            if scores[i] > 0:
                boost = 1.0 + (0.25 if id(ch) in top_ids else 0.0)
                cand.append((scores[i] * boost, self.chunks.index(ch), pos, s, ch))
        best = sorted(cand, key=lambda x: -x[0])[:n]
        if best:  # drop weak extras: keep sentences scoring at least 60% of the best one
            best = [b for b in best if b[0] >= 0.6 * best[0][0]]
        return [(s, ch) for _, _, _, s, ch in sorted(best, key=lambda x: (x[1], x[2]))]

    def _extractive(self, question, hits):
        """Answer from ONE section (the best match), never mixing text from different sections.
        Short sections are returned whole; long ones are trimmed to their most relevant sentences."""
        chunk = hits[0][0]
        if len(chunk["text"]) <= 900:
            return chunk["text"]
        picked = self._pick(question, [(chunk, hits[0][1])], n=4)
        return " ".join(s for s, _ in picked) or chunk["text"][:900]

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
        content = [w for w in tokens(search_q) if len(w) > 2]
        known_chars = sum(len(w) for w in content if w in known)
        coverage = known_chars / max(1, sum(len(w) for w in content))  # how much of the question the pages know
        if self.best_tfidf < MIN_SCORE or coverage < 0.5:
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
        if PERSONAL.search(question) and re.search(r"lump|nipple|discharge|pain|swell|change|dimpl", question, re.I):
            bc = next(c for c in self.chunks if c["heading"] == "When to see a doctor")
            hits = [(bc, 1.0)] + [h for h in hits if h[0] is not bc]
        if PERSONAL.search(question):
            answer = ("I can't give personal medical advice. NCI says to contact your doctor if you notice "
                      "any changes in your breasts. Here is the general information: ") + answer
        seen, sources = set(), []
        used = [hits[0][0]]
        if PERSONAL.search(question) and hits[0][0]["heading"] == "When to see a doctor":
            used = [hits[0][0]] + used
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
