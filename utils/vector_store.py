import os
import re
import shutil

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_PERSIST_DIRECTORY = "chroma_db"
DEFAULT_COLLECTION_PREFIX = "repo"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_session_id(session_id):
    """
    Collection names and folder names should only contain safe characters.
    Falls back to 'default' if nothing usable is passed (keeps old
    single-user behavior working if this is ever called without a session_id).
    """
    if not session_id:
        return "default"
    return _SAFE_ID_PATTERN.sub("_", str(session_id))


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """
    Returns the embedding model, cached for the lifetime of the server
    process. Without this, every single repo processed re-loads the model
    weights from disk/cache, which is slow and wasteful — one repo per
    session no longer needs its own copy of the model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(chunks, session_id=None):
    """
    Creates a fresh Chroma vector database scoped to a single session_id,
    so concurrent users never read/write each other's data or wipe each
    other's index.
    """

    safe_id = _sanitize_session_id(session_id)
    persist_directory = os.path.join(BASE_PERSIST_DIRECTORY, safe_id)
    collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{safe_id}"

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embeddings = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    return vector_db


def load_vector_store(session_id=None):
    """
    Loads an existing vector database for a specific session_id.
    Useful when we don't want to re-index the repository every time.
    """

    safe_id = _sanitize_session_id(session_id)
    persist_directory = os.path.join(BASE_PERSIST_DIRECTORY, safe_id)
    collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{safe_id}"

    embeddings = get_embedding_model()

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    return vector_db


def delete_vector_store(session_id):
    """
    Removes a session's vector store from disk. Call this when a session
    ends or is cleared, so cloned repos + embeddings don't pile up forever.
    """
    safe_id = _sanitize_session_id(session_id)
    persist_directory = os.path.join(BASE_PERSIST_DIRECTORY, safe_id)

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)