import os
import shutil
import time
import uuid
from collections import deque

import streamlit as st
from groq import RateLimitError, AuthenticationError, PermissionDeniedError, NotFoundError, APIConnectionError, APIStatusError
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

# Personal credentials stay in this browser session; never change os.environ.
SERVER_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
needs_personal_key = st.session_state.get("shared_key_limited", False) or not SERVER_GROQ_API_KEY
if needs_personal_key:
    if st.session_state.get("shared_key_limited"):
        st.sidebar.warning("The shared API usage allowance has been reached.")
    st.sidebar.info("Enter your own Groq API key to continue. Your Groq account's limits still apply.")
    st.sidebar.text_input("Your Groq API key", type="password", key="personal_groq_key")
    st.sidebar.caption("Used only for this session. Reset Session removes it.")
personal_key = st.session_state.get("personal_groq_key", "").strip()
groq_api_key = personal_key or (None if needs_personal_key else SERVER_GROQ_API_KEY)
using_personal_key = bool(personal_key)

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
    st.session_state.pop("pending_question", None)
    st.rerun()

if st.sidebar.button("Reset Session (clear repo + chat)"):
    _cleanup_session_data()
    for key in ("vector_db", "documents", "chunks", "repo_url", "vector_store_id",
                "personal_groq_key", "shared_key_limited", "pending_question", "request_timestamps"):
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
                st.session_state.pop("pending_question", None)

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

retry = False
if st.session_state.get("pending_question"):
    st.info("Your previous question is saved. You can retry it after entering a key or waiting for quota to recover.")
    retry = st.button("Retry previous question", disabled=not bool(groq_api_key))
new_question = st.chat_input("Ask question about the codebase")
user_question = st.session_state.get("pending_question") if retry else new_question

if user_question:
    if not retry:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
    st.session_state.pending_question = user_question

    if "vector_db" not in st.session_state:
        st.warning("Please process a repository first.")
    elif not groq_api_key:
        st.warning("Please enter Groq API Key.")
    elif not using_personal_key and not _check_rate_limit():
        st.warning(
            f"You've hit the rate limit ({RATE_LIMIT_MAX_REQUESTS} questions "
            f"per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes). Please wait a bit "
            "or enter your own key to continue."
        )
        st.session_state.shared_key_limited = True
        st.rerun()
    else:
        try:
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

        except RateLimitError:
            if not using_personal_key:
                st.session_state.shared_key_limited = True
                st.rerun()
            st.warning("Your Groq key has reached its usage limit. Wait for quota to recover or replace it with a key from an account with available quota, then retry.")
            st.stop()
        except AuthenticationError:
            st.error("Groq rejected the API key. Check your session key and retry." if using_personal_key else "The shared API key was rejected. The app owner needs to check its configuration.")
            st.stop()
        except (PermissionDeniedError, NotFoundError):
            st.error("This Groq account cannot access the configured model. Check model access or ask the app owner to update GROQ_MODEL.")
            st.stop()
        except APIConnectionError:
            st.warning("Could not connect to Groq. Please retry shortly.")
            st.stop()
        except APIStatusError:
            st.warning("Groq could not complete the request. Please retry shortly.")
            st.stop()

        st.session_state.pop("pending_question", None)
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