import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIRECTORY = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Returns the embedding model.
    Keeping this separate makes it easy to switch
    to another embedding model later.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(chunks):
    """
    Creates a fresh Chroma vector database.
    """

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    embeddings = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="github_codebase"
    )

    return vector_db


def load_vector_store():
    """
    Loads an existing vector database.
    Useful when we don't want to re-index
    the repository every time.
    """

    embeddings = get_embedding_model()

    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="github_codebase"
    )

    return vector_db