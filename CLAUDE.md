# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app (`app.py`) that lets a user paste a public GitHub repo URL, clones and indexes it, and then answers natural-language questions about the codebase using a RAG pipeline backed by Groq (Llama 3.3 70B Versatile) and ChromaDB.

## Commands

Run the app:
```
streamlit run app.py
```

Run evals (each is a standalone script, never imported by `app.py`, with zero effect on live app behavior):
```
python -m eval.evaluate_router        # tests utils/query_router.py's route_question() against eval/golden_router_questions.py
python -m eval.evaluate_correctness   # runs the full pipeline against eval/golden_correctness_questions.py, judged by a separate LLM call
```
Both require `GROQ_API_KEY` in `.env` or the environment, and both write a CSV of results next to the script (`eval/router_eval_results.csv`, `eval/correctness_eval_results.csv`). The correctness eval clones its own throwaway copy of the repo under `repos/correctness_eval/` and deletes its vector store when done — it's isolated from real user sessions. The correctness golden set targets this repo's own README/`repo_loader.py`, so update `eval/golden_correctness_questions.py` if that content changes.

There is no lint/test/build tooling configured (no pytest, no linter config) — the `eval/` scripts above are the only form of automated verification in this repo.

Required env vars (`.env`, loaded via `python-dotenv`): `GROQ_API_KEY` (Groq LLM access — used server-side if set; otherwise the Streamlit sidebar asks each user for their own key), `HF_TOKEN` (Hugging Face, for the embedding/reranker models).

## Architecture

The pipeline is: clone repo → load & tag files → chunk → embed → store in Chroma → (per question) rewrite → route → multi-retrieve → dedupe → rerank → generate. Each stage is its own module under `utils/`, wired together by `app.py` (UI + orchestration) and `utils/rag_chain.py` (the per-question RAG call). Full stage-by-stage walkthrough is in README.md's "Workflow" section — read it before touching the pipeline.

**Multi-session isolation is the load-bearing design constraint.** Every session gets a `session_id` (`st.session_state.session_id`, a UUID), and that ID scopes *everything* per-user: the clone path (`repos/<session_id>/cloned_repo`), the Chroma persist directory and collection name (`chroma_db/<session_id>/`, via `_sanitize_session_id` in `utils/vector_store.py`), and the vector-store-id used for cleanup. Concurrent users on the same server process never read or write each other's data. When adding new per-repo state, thread it through `session_id` the same way rather than using a shared/global path. Each successful "Process Repository" run also mints a *new* vector-store id (`session_id_<uuid8>`) rather than reusing the old one, because Chroma caches DB clients by path and reusing a path after a delete causes "readonly database" errors — see the comment in `app.py` around `create_vector_store` for why.

**`utils/repo_loader.py` is the security boundary.** `clone_github_repo` only accepts `https://github.com/<owner>/<repo>` URLs (regex-validated), checks repo size via the GitHub API *before* cloning (rejects >150MB, private, or nonexistent repos), and clones with `depth=1` as defense-in-depth. Any change to URL handling or cloning needs to preserve this validate-then-check-size-then-clone ordering — `check_repo_size` exists specifically so an oversized clone is never attempted in the first place.

**File classification drives retrieval.** `repo_loader.py`'s `get_file_type()` and `get_file_priority()` tag every loaded file with a `file_type` (readme/documentation/source_code/test/configuration/dependency/license/repo_structure) and a numeric priority. These metadata fields are the join key used throughout the rest of the pipeline:
- `utils/code_splitter.py` picks a chunking strategy (code-aware, markdown-aware, or large-block structure splitter) based on `file_type`/extension.
- `utils/multi_retriever.py`'s `multi_retrieve()` filters Chroma queries by `file_type` per question category (e.g. `architecture` pulls from repo_structure + readme + docs; `implementation` pulls from source_code + docs + readme).
- `utils/rag_chain.py`'s final sort and the reranker both use `file_priority` as a tiebreaker/signal alongside the cross-encoder score.

If you add a new file type or category, you generally need to update all three of these files together, plus `utils/query_router.py`'s `VALID_CATEGORIES` list and prompt (the category taxonomy is duplicated as a hardcoded list there and must stay in sync with what `multi_retrieve()` handles).

**Question flow before the RAG chain even runs:** `app.py` first calls `utils/question_rewriter.py`'s `rewrite_question()` (turns conversational follow-ups like "explain that simply" into standalone questions, using the last 6 chat messages — no-ops if there's no real history) and then passes the rewritten question into `create_rag_chain()`. Inside `create_rag_chain` (`utils/rag_chain.py`), the question is routed to a category (`utils/query_router.py`), multi-retrieved (`utils/multi_retriever.py`, capped at 20 docs), deduplicated, then reranked by a cross-encoder (`utils/reranker.py`, cached process-wide via `st.cache_resource`, top 6 survive) before being stuffed into the final answer-generation prompt. Both the embedding model and reranker model are `@st.cache_resource`-cached at module scope — loading them is expensive, and this cache is what makes multiple sessions share one copy instead of reloading weights per session.

**License retrieval has a fallback**: `retrieve_license()` in `multi_retriever.py` falls back to README chunks if no dedicated LICENSE file matched, since many repos state their license only in the README.

**`utils/rag_chain.py` has a large dead code block** (two `'''...'''`-quoted earlier versions of `create_rag_chain`) left at the bottom of the file after the string-matching router was replaced by the LLM-based one — be aware it's inert, not a second code path.

## Data directories

`repos/` and `chroma_db/` are runtime-generated per-session working directories (cloned repos and vector stores), not source — sessions clean up after themselves via `_cleanup_session_data()` in `app.py` and `delete_vector_store()` in `utils/vector_store.py`, but stale directories from crashed/abandoned sessions can accumulate.
