# 💻 GitHub Codebase Assistant

An intelligent Conversational Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about any public GitHub repository. The application clones a repository, indexes its source code and documentation, retrieves and **reranks** the most relevant chunks, and provides accurate, source-grounded answers using Large Language Models (LLMs).

This was developed by **YAKKALA LAKSHMI CHARAN**, a student from RGUKT Nuzvid.

---

## 🚀 Features

- Clone and process any public GitHub repository
- Intelligent code and documentation indexing
- Code-aware and documentation-aware chunking
- Hugging Face Embeddings for semantic search
- ChromaDB Vector Database
- LLM-powered Query Routing
- Conversational Question Rewriting
- Multi-Retriever Architecture
- **Cross-Encoder Reranking of retrieved chunks**
- Auto-generated Repository Structure
- Source-aware Answers with File References
- Interactive Streamlit Chat Interface

---

## 🏗️ System Architecture

```
            GitHub Repository URL
                      │
                      ▼
             Clone Repository
                      │
                      ▼
            Load Repository Files
                      │
                      ▼
        Generate Repository Structure
                      │
                      ▼
      Code & Documentation Chunking
                      │
                      ▼
          Generate Vector Embeddings
                      │
                      ▼
             Store in ChromaDB
                      │
                      ▼
             User Asks Question
                      │
                      ▼
         Question Rewriting (LLM)
                      │
                      ▼
           Query Router (LLM)
                      │
                      ▼
          Multi-Retriever Search
                      │
     ┌──────────┬──────────┬──────────┐
     │          │          │          │
     ▼          ▼          ▼          ▼
 README      Source     Docs      Repo Structure
Retriever   Retriever Retriever    Retriever
     │          │          │          │
     └──────────┴──────────┴──────────┘
                      │
                      ▼
          Merge Retrieved Documents
                      │
                      ▼
          Remove Duplicate Chunks
                      │
                      ▼
        Cross-Encoder Reranking
                      │
                      ▼
            Sort by Reranker Score
             (with File Priority)
                      │
                      ▼
           Groq LLM Generates Answer
                      │
                      ▼
          Answer with Source Citations
```

---

## 📂 Project Structure

```
github-codebase-assistant/
│
├── app.py
│
├── utils/
│   ├── repo_loader.py
│   ├── code_splitter.py
│   ├── vector_store.py
│   ├── query_router.py
│   ├── question_rewriter.py
│   ├── multi_retriever.py
│   ├── reranker.py          # cross-encoder reranking of retrieved chunks
│   └── rag_chain.py
│
├── chroma_db/
│
├── repos/
│
├── requirements.txt
│
└── README.md
```

> Note: `reranker.py` is listed based on the reranking behavior visible in `app.py` (each source chunk now carries a `reranker_score`). If you named the module differently, update this section accordingly.

---

## ⚙️ Tech Stack

**Programming Language**
- Python

**LLM**
- Groq
- Llama 3.3 70B Versatile

**Frameworks**
- LangChain
- Streamlit

**Vector Database**
- ChromaDB

**Embedding Model**
- sentence-transformers/all-MiniLM-L6-v2

**Reranking Model**
- Cross-Encoder via `sentence-transformers` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- *(confirm/replace with the exact model you're using)*

**Version Control**
- Git
- GitHub

---

## 🔄 Workflow

**Step 1 — Repository Input**
The user enters a public GitHub repository URL.
Example: `https://github.com/pallets/flask`

**Step 2 — Clone**
The repository is cloned locally.

**Step 3 — File Scanning**
The application scans all supported files including Python, JavaScript, TypeScript, Java, Markdown, YAML, JSON, and TOML — while ignoring unnecessary folders like `.git`, `node_modules`, `venv`, `build`, and `dist`.

**Step 4 — Metadata Enrichment**
Each file is enriched with metadata such as File Name, File Type, File Extension, and File Priority.

**Step 5 — Repository Structure Generation**
A repository structure document is automatically generated to improve architecture-related queries.

**Step 6 — Chunking**
Files are split intelligently. Source code and documentation use different chunking strategies to preserve semantic meaning.

**Step 7 — Embedding**
Each chunk is converted into vector embeddings using Hugging Face Embeddings.

**Step 8 — Storage**
The embeddings are stored in ChromaDB.

**Step 9 — User Question**
The user asks a question. Example: *"How are routes implemented?"*

**Step 10 — Question Rewriting**
If the question is conversational, the Question Rewriter converts it into a standalone question.
Before: *"Explain that simply."*
After: *"Explain how routes are implemented in simple words."*

**Step 11 — Query Routing**
The Query Router classifies the question into one of the following categories: Overview, Architecture, Implementation, Testing, Configuration, Dependency, License, General.

**Step 12 — Multi-Retriever Search**
The Multi-Retriever retrieves relevant information from multiple sources depending on the question category.
- Architecture questions retrieve from: Repository Structure, README, Documentation
- Implementation questions retrieve from: Source Code, Documentation

**Step 13 — Merge & Deduplicate**
The retrieved documents are merged and deduplicated.

**Step 14 — Reranking**
Each surviving chunk is scored by a **cross-encoder reranker** against the (rewritten) question, producing a `reranker_score`. This re-orders results by actual relevance rather than raw vector-similarity alone, correcting cases where embedding search alone would surface a superficially similar but less useful chunk.

**Step 15 — Priority-Aware Sorting**
The reranked documents are sorted using a combination of reranker score and file priority before being passed to the LLM.

**Step 16 — Answer Generation**
The Groq LLM (Llama 3.3 70B Versatile) generates a response using only the retrieved, reranked repository context, and returns a source-aware answer.

---

## ✨ Example Questions

**Overview**
- What is Flask?
- What problem does this project solve?
- Explain this project.

**Architecture**
- Explain the project architecture.
- Explain the folder structure.
- What are the main modules?

**Implementation**
- How are routes implemented?
- Where is Session defined?
- Explain this function.

**Configuration**
- How is the project configured?
- Where are environment variables defined?

**Dependency**
- Which libraries are required?
- What packages does this project use?

---

## 🌟 Key Features

- Conversational RAG
- Semantic Search
- LLM-based Query Routing
- Multi-Retriever Architecture
- **Cross-Encoder Reranking**
- Repository Structure Generation
- Source-aware Responses
- Metadata-aware Retrieval
- Duplicate Removal
- Priority + Relevance-based Context Selection

---

## 📈 Future Enhancements

- Parent-Child Retrieval
- Hybrid Search (BM25 + Vector Search)
- Repository Summarization
- Mermaid Architecture Diagram Generation
- Cross-file Execution Tracing
- Multi-Agent Code Analysis
- Support for Private GitHub Repositories
- Code Review and Bug Detection

---

## 🚀 Deployment

### Environment variables

Copy `.env.example` to `.env` and fill in real values (never commit `.env` — it's gitignored):

- `GROQ_API_KEY` — used server-side for all LLM calls. If set, users never have to paste their own key. If omitted, each user is prompted for their own key in the sidebar.
- `HF_TOKEN` — only needed if you're gated/rate-limited on Hugging Face for the embedding/reranker model downloads.

### Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

### Docker

A `Dockerfile` is included (CPU-only, based on `python:3.11-slim`). This is the most portable option and works on Render, Railway, Fly.io, Google Cloud Run, Hugging Face Spaces (Docker SDK), or any container host:

```
docker build -t github-codebase-assistant .
docker run -p 8501:8501 --env-file .env github-codebase-assistant
```

Then open http://localhost:8501.

### Streamlit Community Cloud

Point Streamlit Cloud at this repo/`app.py` directly — `packages.txt` (system `git`) and `runtime.txt` (Python version) are picked up automatically. Set `GROQ_API_KEY` (and `HF_TOKEN` if needed) as app secrets rather than in `.env`.

### Other PaaS (Railway, Render, Heroku-style buildpacks)

A `Procfile` is included for platforms that build from `requirements.txt` rather than a Dockerfile.

### Notes

- `repos/` and `chroma_db/` are per-session scratch directories written at runtime (see `CLAUDE.md`) — they don't need to persist across restarts, so ephemeral container storage is fine.
- The app needs a working `git` binary on the host/container (used by `GitPython` to clone repos) — the Dockerfile and `packages.txt` both install it.
- Embeddings/reranker models are CPU-only by default (`requirements.txt` pins `torch` from the CPU wheel index) to keep image size and cold-start time reasonable on typical PaaS free/hobby tiers.

---

## 👨‍💻 Author

**Lakshmi Charan Yakkala**
- GitHub: https://github.com/lakshmicharan18
- LinkedIn: https://www.linkedin.com/in/charan-yakkala-95bbb8318/

---

## 📄 License

This project is intended for educational and learning purposes.
