import os
import shutil
import time
import uuid
from collections import deque

import streamlit as st
from dotenv import load_dotenv

from utils.repo_loader import clone_github_repo, load_code_files, RepoValidationError
from utils.code_splitter import split_code_files
from utils.vector_store import create_vector_store, delete_vector_store
from utils.rag_chain import create_rag_chain
from utils.question_rewriter import rewrite_question

load_dotenv()

st.set_page_config(page_title="GitHub Codebase Assistant", page_icon="💻")

st.title("💻 GitHub Codebase Assistant")
st.write("Enter a GitHub repository URL and ask questions about the codebase.")

# Each browser session gets its own isolated repo clone + vector store,
# so concurrent users never overwrite or read each other's data.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

session_id = st.session_state.session_id
session_repo_path = os.path.join("repos", session_id, "cloned_repo")

# In production, set GROQ_API_KEY as a server-side environment variable so
# users never have to paste their own key into the browser. The sidebar
# input is kept only as a fallback for local development when no server
# key is configured.
SERVER_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if SERVER_GROQ_API_KEY:
    groq_api_key = SERVER_GROQ_API_KEY
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    st.sidebar.caption(
        "No server-side GROQ_API_KEY is configured — using your own key "
        "for this session only."
    )

repo_url = st.text_input("Enter GitHub Repo URL")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Simple per-session rate limiting on chat questions ---
# Protects a shared server-side Groq key from being hammered by one user.
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 600  # 10 minutes


def _check_rate_limit():
    now = time.time()

    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = deque()

    timestamps = st.session_state.request_timestamps

    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    timestamps.append(now)
    return True


def _cleanup_session_data():
    """Removes this session's cloned repo and vector store from disk."""
    if os.path.exists(session_repo_path):
        shutil.rmtree(os.path.dirname(session_repo_path), ignore_errors=True)
    old_store_id = st.session_state.get("vector_store_id")
    if old_store_id:
        delete_vector_store(old_store_id)


if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("Reset Session (clear repo + chat)"):
    _cleanup_session_data()
    for key in ("vector_db", "documents", "chunks", "repo_url", "vector_store_id"):
        st.session_state.pop(key, None)
    st.session_state.messages = []
    st.rerun()

if st.button("Process Repository"):
    if not repo_url:
        st.warning("Please enter a GitHub repo URL")
    else:
        try:
            with st.spinner("Processing repository..."):
                repo_path = clone_github_repo(repo_url, repo_path=session_repo_path)
                documents = load_code_files(repo_path)
                chunks = split_code_files(documents)

                # Every successful processing run gets its own fresh vector
                # store path (never reused within this running process).
                # Chroma caches DB clients by path internally, so reusing a
                # path after deleting it (e.g. after Reset) causes
                # "readonly database" errors — a brand new path sidesteps
                # that entirely.
                old_store_id = st.session_state.get("vector_store_id")
                new_store_id = f"{session_id}_{uuid.uuid4().hex[:8]}"

                vector_db = create_vector_store(chunks, session_id=new_store_id)

                if old_store_id:
                    delete_vector_store(old_store_id)

                st.session_state.vector_store_id = new_store_id
                st.session_state.vector_db = vector_db
                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.repo_url = repo_url
                st.session_state.messages = []

            st.success("Repository indexed successfully!")
            st.write("Total files loaded:", len(documents))
            st.write("Total chunks created:", len(chunks))

            with st.expander("View loaded files"):
                for doc in documents[:30]:
                    st.write(
                        doc.metadata.get("source"),
                        "|",
                        doc.metadata.get("file_type")
                    )
        except RepoValidationError as e:
            st.error(str(e))

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_question = st.chat_input("Ask question about the codebase")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    st.chat_message("user").write(user_question)

    if "vector_db" not in st.session_state:
        st.warning("Please process a repository first.")
    elif not groq_api_key:
        st.warning("Please enter Groq API Key.")
    elif not _check_rate_limit():
        st.warning(
            f"You've hit the rate limit ({RATE_LIMIT_MAX_REQUESTS} questions "
            f"per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes). Please wait a bit "
            "before asking another question."
        )
    else:
        with st.spinner("Generating answer..."):
            standalone_question = rewrite_question(
                user_question,
                st.session_state.messages,
                groq_api_key
            )

            chat_history_text = ""

            for msg in st.session_state.messages[-6:]:
                chat_history_text += f"{msg['role']}: {msg['content']}\n"

            chain, category, retrieved_docs, context_text = create_rag_chain(
                st.session_state.vector_db,
                groq_api_key,
                standalone_question,
                chat_history_text
            )

            response = chain.invoke({
                "input": standalone_question,
                "category": category,
                "chat_history": chat_history_text,
                "context": context_text
            })

            answer = response.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)

            st.write("### Question Category")
            st.info(category)

            st.write("### Rewritten Question")
            st.info(standalone_question)

            with st.expander("Source Code Chunks Used"):
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"### Source {i + 1}")
                    st.write("📄 File:", doc.metadata.get("source"))
                    st.write("📂 Type:", doc.metadata.get("file_type"))
                    st.write("⭐ Priority:", doc.metadata.get("file_priority"))
                    st.write("Reranker Score:", round(doc.metadata.get("reranker_score", 0), 4))
                    st.code(doc.page_content[:1200])
                    st.write("--------")