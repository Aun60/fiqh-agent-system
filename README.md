# Fiqh Multi-Agent RAG System

A multi-agent RAG system that answers Islamic fiqh (jurisprudence) questions
grounded in classical texts from the four Sunni madhabs, Hanafi, Shafi'i,
Maliki, and Hanbali. Every ruling is attributed to its madhab, scholar, and
source text, and the system is built specifically to never blend or mix up
positions between schools.

## What makes this "multi-agent" and not just RAG

Four agents work in a pipeline for every question:

1. **Router agent**, classifies the question as `single` (one madhab
   explicitly named), `comparative` (no madhab named, answer from all
   four), or `off_topic` (not a fiqh question). Defaults to `comparative`
   on any ambiguity rather than silently guessing a madhab.
2. **Retriever agent**, queries ChromaDB with a **hard metadata filter**
   on `madhab`. In single mode, chunks from other madhabs are never even
   fetched from the database, not just ignored downstream.
3. **Synthesizer agent**, writes the answer, required to attribute every
   ruling to its madhab, scholar, and source text, and to keep each
   madhab's position in comparative mode in its own clearly labeled
   section, never merged into one paragraph.
4. **Verifier agent**, re-checks the draft against the actually retrieved
   passages before it reaches the user. Catches hallucinated attributions
   (a ruling claimed from a source that doesn't actually say that) and
   madhab leaks (an answer scoped to one school mentioning another). On a
   failure, the specific issue is fed back to the synthesizer for a
   corrected retry (up to 3 attempts) before falling back to a safe
   "consult a scholar" message rather than showing a risky answer.

This last step matters in testing: the verifier has caught and corrected
at least one real hallucination during development (an invented hadith
citation attributed to a Hanbali source that didn't contain it), proof
the safety layer isn't just theoretical.

## Corpus coverage (honest state, as of this build)

The knowledge base currently has meaningfully more depth for **Hanafi and
Shafi'i** than for **Maliki and Hanbali**, a function of which classical
texts were sourced, not a bug in the pipeline. When Maliki or Hanbali
sources don't directly address a question, the system says so explicitly
("the retrieved [Madhab] sources don't directly address this, a scholar
from that school should be consulted") rather than guessing or staying
silent. Expanding the Maliki/Hanbali corpus with more focused texts is the
clearest next improvement, see "Possible next steps" below.

## Architecture

```
fiqh-agent-system/
├── data/raw_pdfs/              # source PDFs (not committed; see .gitignore)
├── ingestion/
│   ├── source_registry.py       # maps each PDF filename -> madhab/scholar/text
│   │                             # (the single source of truth for attribution)
│   ├── loader.py                  # PDF -> text, with OCR fallback for scanned PDFs
│   ├── chunker.py                  # paragraph-aware chunking with overlap
│   ├── metadata_tagger.py           # stamps madhab/scholar/text onto every chunk
│   └── embed_and_store.py            # embeds chunks (sentence-transformers) -> ChromaDB
├── agents/
│   ├── router_agent.py            # single / comparative / off_topic classification
│   ├── retriever_agent.py          # hard-filtered Chroma queries by madhab
│   ├── synthesizer_agent.py         # writes the attributed answer
│   ├── verifier_agent.py             # catches hallucinations & madhab leaks
│   └── llm_client.py                  # shared Gemini API wrapper (all agents call this)
├── orchestrator.py               # wires router -> retriever -> synthesizer -> verifier
├── api/main.py                    # FastAPI endpoint (POST /query, GET /health)
├── frontend/index.html             # single-file landing page + full-screen chat UI
├── requirements.txt
├── .env.example
└── .gitignore
```

## Tech stack

- **LLM:** Google Gemini (`gemini-3.1-flash-lite`), free tier, no card required
- **Vector DB:** ChromaDB, persisted locally in `chroma_db/`
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), local and free
- **Backend:** FastAPI
- **Frontend:** single-file HTML/CSS/JS, no build step, no framework
- **PDF processing:** `pypdf` for text-layer PDFs, with `pytesseract` +
  `pdf2image` (Tesseract OCR) as a fallback for scanned/image-only PDFs

## Setup

### 1. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install OCR tools (needed if any of your source PDFs are scanned images)
- **Tesseract:** https://github.com/UB-Mannheim/tesseract/wiki (Windows installer),
  `brew install tesseract` (Mac), `sudo apt install tesseract-ocr` (Linux)
- **Poppler:** https://github.com/oschwartz10612/poppler-windows/releases (Windows;
  unzip and add the `Library/bin` folder to PATH), `brew install poppler` (Mac),
  `sudo apt install poppler-utils` (Linux)

### 3. Get a free Gemini API key
Go to https://aistudio.google.com/apikey, sign in, click "Create API key" (no billing required for the free tier). Create a `.env` file in the project
root (copy `.env.example`):
```
GEMINI_API_KEY=your_key_here
```

### 4. Add your source PDFs
Place them in `data/raw_pdfs/`. Then open `ingestion/source_registry.py`
and make sure every filename is mapped to its correct `madhab`, `scholar`,
and `text_title`. This file is the single source of truth for attribution;
nothing else in the system guesses a madhab from content.

### 5. Ingest into ChromaDB
```bash
python -m ingestion.embed_and_store
```
Prints per-madhab chunk counts when done; use this to sanity-check
coverage before querying.

### 6. Run the backend
```bash
uvicorn api.main:app --reload
```
Confirm it's up at http://127.0.0.1:8000/health (should return `{"status":"ok"}`).

### 7. Open the frontend
Open `frontend/index.html` directly in a browser, or serve it locally:
```bash
cd frontend
python -m http.server 5500
```
then visit http://127.0.0.1:5500. Click "Let's Query about Fiqh Questions"
to enter the chat.

## Example questions to try

**Single-madhab** (should return one colored tag, one school's ruling):
- "What is the Hanafi ruling on the amount of water needed for wudu to be valid?"
- "According to the Shafi'i school, what breaks a fast during Ramadan?"
- "In the Maliki madhab, what is required for a valid Friday (Jumu'ah) prayer?"

**Comparative** (no madhab named, should return four separate labeled sections):
- "Can prayers be combined while traveling?"
- "Is it obligatory to recite Surah al-Fatiha behind an imam in congregational prayer?"
- "What breaks the fast during Ramadan?"

## Known limitations

- **Corpus depth is uneven across madhabs** (see "Corpus coverage" above);
  Maliki and Hanbali answers more often come back as "sources don't
  directly address this."
- **OCR-derived text quality varies**, a few source PDFs were scanned
  images rather than digitally typeset text, so their extracted text (and
  therefore retrieval quality) is somewhat noisier than the
  digitally-native sources.
- **Free-tier LLM rate limits**, Gemini's free tier has requests-per-minute
  caps; the pipeline makes 2-4 model calls per question (router,
  synthesizer, up to 2 retries, verifier), so rapid-fire testing can
  occasionally hit a rate limit. The client retries automatically with
  backoff.
- **Not a fatwa service**, this draws only from the specific classical
  texts in the corpus and is explicitly framed as educational; every
  answer includes a reminder to consult a qualified scholar for personal
  or complex rulings.

## Deployment notes

- **Backend on Render (free tier) is untested/risky at this size**: Chroma
  + `sentence-transformers` (which pulls in PyTorch) is a heavy combination
  for Render's free 512MB RAM tier, and Render's free tier disk is
  ephemeral (wiped on redeploy), so a locally-built `chroma_db/` folder
  won't persist there without a paid persistent disk add-on.
- **Frontend on Vercel**: straightforward, since `frontend/index.html` is
  a static file with no build step. Just point `API_URL` near the top of
  its `<script>` tag at your deployed backend's URL before deploying.
- If the free-tier backend deploy doesn't work as-is, the two most direct
  fixes are: (1) swap local `sentence-transformers` embeddings for a
  hosted embeddings API (removes the PyTorch weight from the deployed
  backend entirely), and (2) move the vector store to a hosted option
  (Chroma Cloud, or a persistent-disk add-on) instead of a local folder.

## Possible next steps

- Expand the Maliki and Hanbali corpus with more focused primary texts
- Add conversation memory (follow-up questions referencing earlier answers)
- Add a citation view (click a ruling to see the exact retrieved passage
  it's grounded in)
- Move from a single-file frontend to a proper React app if the project
  grows further