# 🔍 Multimodal RAG — Local, Private, No API Key Required

A fully local Retrieval-Augmented Generation (RAG) pipeline that understands **text, tables, and images** inside PDF documents — powered by Ollama, LangChain, and ChromaDB. No OpenAI. No API keys. No data leaving your machine.

---

## 🧠 Why This Is Different From Most RAG Projects

Most "chat with PDF" projects only extract plain text. Real-world documents — research papers, financial reports, HR policies, clinical guidelines — contain:

- **Tables** with structured data (salary bands, statistics, comparisons)
- **Figures and charts** (trend lines, regional breakdowns, diagrams)
- **Mixed layouts** where a table's meaning depends on the text surrounding it

This pipeline handles all three, using a vision-capable local LLM to describe images and a context-aware chunking strategy that preserves narrative context around tables.

It also goes beyond basic vector search — implementing **hybrid search (BM25 + semantic + RRF)** so exact terms, numbers, and acronyms are retrieved as reliably as conceptual matches.

---

## 🏗️ Pipeline Architecture

```
📁 PDF Documents (multiple files)
         │
         ▼
┌─────────────────────────┐
│   1. PARTITION (hi_res) │  ← unstructured + Tesseract OCR
│   Extract text, tables, │    Layout detection via HuggingFace
│   images from each page │    model (runs locally)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   2. CHUNK BY TITLE     │  ← Title-aware chunking
│   Tables converted to   │    Context from neighboring chunks
│   CompositeElement with │    preserved around tables
│   surrounding context   │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   3. SEPARATE CONTENT   │  ← Per-chunk content analysis
│   text / tables / imgs  │    Images stored as base64
│   extracted per chunk   │    Tables stored as HTML
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   4. AI SUMMARIZATION   │  ← ChatOllama (llama3.2-vision)
│   Each chunk gets an    │    Describes images, interprets
│   enhanced searchable   │    tables, merges with text
│   description           │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   5. EMBED + STORE      │  ← nomic-embed-text (local)
│   Cosine similarity     │    ChromaDB persisted to disk
│   vector store          │
└─────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│   6. HYBRID RETRIEVAL (NEW)          │
│   Vector search (semantic)    ──┐    │
│   BM25 keyword search         ──┼──► │  RRF merges both
│   Reciprocal Rank Fusion (RRF)  │    │  ranked lists
└──────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   7. ANSWER GENERATION  │  ← ChatOllama generates answer
│   User query → chunks   │    strictly from retrieved context
│   → LLM final answer    │
└─────────────────────────┘
```

---

## 📂 Project Structure

```
multimodal-rag-local/
├── notebooks/
│   └── multimodal_rag.ipynb       # Main pipeline notebook
├── docs/                          # Your PDF documents go here
│   ├── WHO_Health_Statistics_2024.pdf
│   ├── TechMeka_HR_Policy.pdf
│   ├── attention_is_all_you_need.pdf
│   └── ...
├── db/
│   └── chroma_db/                 # Auto-generated vector store
├── outputs/
│   └── rag_results.json           # Sample query results
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Parsing | `unstructured[all-docs]` | Extracts text, tables (HTML), and images (base64) from real PDFs |
| OCR | `tesseract` | Reads text embedded in scanned images |
| Layout Detection | HuggingFace layout model (local) | Visually identifies titles, tables, figures on each page |
| Chunking | `chunk_by_title` | Title-aware chunking that respects document structure |
| Embedding | `nomic-embed-text` via Ollama | Fully local, no API needed |
| Vector Store | ChromaDB | Persisted locally with cosine similarity |
| Keyword Search | BM25 (`rank_bm25`) | Exact term matching for numbers, acronyms, codes |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) | Merges vector + BM25 ranked lists without manual score tuning |
| LLM | `llama3.2` / `llama3.2-vision` via Ollama | Fully local, handles image descriptions |
| Orchestration | LangChain | Retriever, document, and chain abstractions |

---

## 🚀 Setup

### 1. System dependencies (Mac)

```bash
brew install poppler tesseract libmagic libheif
```

### 2. Install Ollama and pull models

```bash
# Install Ollama from https://ollama.com
ollama pull nomic-embed-text     # embedding model
ollama pull llama3.2-vision      # vision-capable LLM
ollama serve                     # start Ollama server
```

### 3. Clone the repo and set up Python environment

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-rag-local.git
cd multimodal-rag-local

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your PDFs

Drop any PDF files into the `docs/` folder.

### 5. Run the notebook

Open `notebooks/multimodal_rag.ipynb` in VS Code or Jupyter and run all cells.

---

## 📋 Requirements

```
unstructured[all-docs]
langchain
langchain-community
langchain-classic
langchain-chroma
langchain-ollama
chromadb
rank_bm25
python-dotenv
```

> **Note:** `langchain-classic` is required for `EnsembleRetriever` (hybrid search fusion). In newer versions of LangChain, it was moved out of the core package.

---

## 🔀 Hybrid Search — How It Works

Standard RAG uses **only vector/semantic search**, which is excellent at finding conceptually related content but can miss exact terms:

```
Query: "What is the HALE figure for 2021?"

Vector search alone → finds chunks about "health trends" and
                      "life expectancy" but may miss the exact
                      table row containing "HALE: 61.9 (2021)"

BM25 alone         → finds chunks with literal "HALE" and "2021"
                      but misses paraphrased or conceptual matches

Hybrid (RRF)       → combines both, surfaces the chunk that scores
                      well in either or both lists ✅
```

This matters most for:
- **Exact numbers and statistics** (`61.9 years`, `$130,000`, `26 weeks`)
- **Acronyms and codes** (`HALE`, `CISSP`, `TM-HR-001`, `401(k)`)
- **Proper names** (`Sarah Mitchell`, `BlueCross BlueShield`)
- **Queries that mix concept and exact term** (`"maternity leave 26 weeks"`)

```python
# How it's implemented in the notebook
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

vector_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
bm25_retriever   = BM25Retriever.from_documents(processed_chunks)
bm25_retriever.k = 5

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]   # equal weight; tune based on your documents
)

results = hybrid_retriever.invoke("Your question here")
```

---

## 📄 Sample Documents Used

| Document | Type | Why Interesting |
|---|---|---|
| WHO World Health Statistics 2024 | Scientific / Public Health | Regional comparison charts, mortality trend lines, statistical tables |
| TechMeka HR Policy Manual | Enterprise Policy | Structured sections, leave entitlement tables, custom figures |
| Attention Is All You Need (2017) | Academic / ML Research | Architecture diagrams, performance comparison tables, dense technical text |

---

## 💬 Example Queries and Answers

**Query:** `"Which WHO region had the highest male-to-female mortality ratio?"`

> Retrieved chunks: WHO Health Statistics 2024 — regional mortality breakdown table
>
> **Answer:** "The European Region had the highest male-to-female mortality ratio at 2.6, followed by the Region of the Americas at 2.3. The African Region had the lowest ratio at 1.4."

---

**Query:** `"How many days of annual leave do TechMeka employees get?"`

> Retrieved chunks: TechMeka HR Policy Manual — Section 3.1 Annual Leave
>
> **Answer:** "All full-time TechMeka employees are entitled to 15 days of paid annual leave per calendar year. Up to 5 days can be carried forward to the following year; any unused leave beyond 5 days is forfeited."

---

**Query:** `"What was the global HALE value in 2021?"` *(exact number — benefits from hybrid search)*

> Retrieved chunks: WHO Statistics — HALE trend table (found via BM25 exact match on "HALE" + "2021")
>
> **Answer:** "Global healthy life expectancy (HALE) dropped to 61.9 years in 2021, a decline of 1.5 years from the 2019 value — back to the level last seen in 2012."

---

## 🔑 Key Design Decisions

**Why local models instead of OpenAI?**
Enterprise and healthcare documents often contain sensitive data. Running everything locally means no data ever leaves your machine — making this suitable for real-world private document use cases.

**Why hybrid search instead of pure vector search?**
Vector embeddings compress text into semantic meaning — great for paraphrased matches, but they blur exact tokens like numbers, codes, and acronyms. BM25 catches what vector search misses. RRF merges both without requiring manual score normalization, since it only uses rank positions (not raw scores) to combine the lists.

**Why context-aware table chunking?**
Standard chunking splits a table from the text that introduces and explains it. This pipeline captures the surrounding narrative context (up to 200 characters before and after) and attaches it to the table chunk — so retrieval brings back not just the numbers but the explanation.

**Why `hi_res` strategy?**
The `hi_res` strategy uses a real document layout model to visually identify elements on the page rather than relying purely on PDF text streams. This correctly identifies figures, charts, and table boundaries that naive text-extraction misses.

---

## 🧪 Running Your Own Queries

After running the ingestion pipeline, query your vector store:

```python
query = "Your question here"

# Hybrid retrieval (vector + BM25 + RRF)
retrieved_chunks = hybrid_retriever.invoke(query)

# Generate answer
answer = generate_final_answer(retrieved_chunks, query)
print(answer)
```

---

## 📌 Limitations

- `hi_res` strategy is slow (~1-3 minutes per PDF page on CPU) — best run once and reuse the persisted ChromaDB
- `llama3.2-vision` requires ~8GB RAM — lighter alternative is `llava:7b`
- Image understanding quality depends on the vision model — complex charts may not be fully interpreted
- Currently supports PDF only — Word docs and PowerPoint require additional unstructured configuration
- BM25 index is built in-memory at query time — for very large document sets, consider a persistent keyword index

---

## 🗺️ Roadmap

- [ ] Add a Gradio/Streamlit chat UI
- [ ] Support `.docx` and `.pptx` input formats
- [ ] Add query history and multi-turn conversation
- [ ] Evaluate retrieval quality with RAGAS metrics
- [ ] Add multi-query retrieval (LLM generates query variations)
- [ ] Role-based document access filtering
- [ ] Docker container for one-command setup

---

## 👤 Author

Built by **Yogesh Meka** as a learning project exploring multimodal RAG with fully local inference.

[GitHub](https://github.com/YOUR_USERNAME)
