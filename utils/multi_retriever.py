import os
import re

from langchain_core.documents import Document
from utils.bm25_retriever import retrieve_keywords, fuse_rankings


# Matches bare filenames the user might type in a question, e.g. "preparation_graph.py"
# or "README.md". Intentionally extension-based rather than tied to ALLOWED_EXTENSIONS
# in repo_loader.py, since the question text won't include a leading path.
FILENAME_MENTION_PATTERN = re.compile(
    r"\b[\w-]+\.(?:py|js|jsx|ts|tsx|java|md|json|yml|yaml|toml|html|css|txt|rst)\b",
    re.IGNORECASE
)


def retrieve_by_filename_mention(vector_db, question, k_per_file=6):
    """
    Category-based filtering alone can starve a question that names a specific
    file (e.g. "explain preparation_graph.py") if that file's chunks don't win
    on embedding similarity within the category's k budget. This does an exact
    metadata lookup instead: if the question mentions a filename that matches
    an indexed document's file_name, pull that file's chunks directly,
    regardless of category.
    """
    mentioned = {m.group(0).lower() for m in FILENAME_MENTION_PATTERN.finditer(question)}

    if not mentioned:
        return []

    try:
        all_metadata = vector_db.get(include=["metadatas"]).get("metadatas", [])
    except Exception:
        return []

    matched_sources = set()
    for meta in all_metadata:
        source = meta.get("source", "")
        file_name = meta.get("file_name") or os.path.basename(source)
        if file_name.lower() in mentioned:
            matched_sources.add(source)

    matched_docs = []
    for source in matched_sources:
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_per_file, "filter": {"source": source}}
        )
        docs = retriever.invoke(question)
        for doc in docs:
            # Lets the reranker guarantee these a spot in the final context
            # even if their score alone wouldn't make the cut — an explicitly
            # named file shouldn't lose an explanation to unrelated chunks
            # that merely score higher.
            doc.metadata["filename_match"] = True
        matched_docs.extend(docs)

    return matched_docs


def get_unique_documents(documents):
    unique_docs = []
    seen = {}

    for doc in documents:
        source = doc.metadata.get("source", "")
        # Compare complete evidence: a shared prefix does not imply that
        # two chunks contain the same implementation. Keep separate files
        # distinct so citations retain their provenance.
        key = (source, doc.page_content)

        if key not in seen:
            seen[key] = doc
            unique_docs.append(doc)
        elif doc.metadata.get("filename_match"):
            seen[key].metadata["filename_match"] = True

    return unique_docs


def retrieve_readme_first(vector_db, question, k=4):
    readme_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"file_type": "readme"}
        }
    )

    return readme_retriever.invoke(question)


def retrieve_documentation(vector_db, question, k=4):
    docs_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"file_type": "documentation"}
        }
    )

    return docs_retriever.invoke(question)


def retrieve_repo_structure(vector_db, question, k=3):
    repo_structure_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"file_type": "repo_structure"}
        }
    )

    return repo_structure_retriever.invoke(question)


def retrieve_source_code(vector_db, question, k=8):
    source_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 30,
            "filter": {"file_type": "source_code"}
        }
    )

    return source_retriever.invoke(question)


def retrieve_tests(vector_db, question, k=8):
    test_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 30,
            "filter": {"file_type": "test"}
        }
    )

    return test_retriever.invoke(question)


def retrieve_configuration(vector_db, question, k=8):
    config_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 30,
            "filter": {"file_type": "configuration"}
        }
    )

    return config_retriever.invoke(question)


def retrieve_dependencies(vector_db, question, k=8):
    dependency_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 30,
            "filter": {"file_type": "dependency"}
        }
    )

    return dependency_retriever.invoke(question)


def retrieve_license(vector_db, question, k=5):
    license_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"file_type": "license"}
        }
    )

    docs = license_retriever.invoke(question)

    # Many repos don't have a dedicated LICENSE file — the license section
    # often just lives inside the README instead. Without this fallback,
    # such repos return zero chunks for license questions even though the
    # answer is right there, and the LLM has no choice but to say
    # "I don't know from this codebase."
    if not docs:
        docs = retrieve_readme_first(vector_db, question, k=k)

    return docs


def retrieve_general(vector_db, question, k=12):
    general_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 40
        }
    )

    return general_retriever.invoke(question)


def sort_documents_by_priority(documents):
    return sorted(
        documents,
        key=lambda doc: doc.metadata.get("file_priority", 10)
    )


def multi_retrieve(vector_db, question, category):
    retrieved_docs = []

    # Run before category-based retrieval, and independent of it: a question
    # that names a specific file should pull that file's chunks regardless
    # of which category the router assigned.
    retrieved_docs.extend(retrieve_by_filename_mention(vector_db, question))

    if category == "architecture":
        retrieved_docs.extend(retrieve_repo_structure(vector_db, question, k=3))
        retrieved_docs.extend(retrieve_readme_first(vector_db, question, k=3))
        retrieved_docs.extend(retrieve_documentation(vector_db, question, k=3))
        retrieved_docs.extend(retrieve_source_code(vector_db, question, k=4))

    elif category == "overview":
        retrieved_docs.extend(retrieve_readme_first(vector_db, question, k=6))
        retrieved_docs.extend(retrieve_documentation(vector_db, question, k=3))

    elif category == "implementation":
        retrieved_docs.extend(retrieve_source_code(vector_db, question, k=8))
        retrieved_docs.extend(retrieve_documentation(vector_db, question, k=3))
        retrieved_docs.extend(retrieve_readme_first(vector_db, question, k=2))

    elif category == "testing":
        retrieved_docs.extend(retrieve_tests(vector_db, question, k=8))
        retrieved_docs.extend(retrieve_documentation(vector_db, question, k=2))

    elif category == "configuration":
        retrieved_docs.extend(retrieve_configuration(vector_db, question, k=8))
        retrieved_docs.extend(retrieve_readme_first(vector_db, question, k=2))

    elif category == "dependency":
        retrieved_docs.extend(retrieve_dependencies(vector_db, question, k=8))
        retrieved_docs.extend(retrieve_readme_first(vector_db, question, k=2))

    elif category == "license":
        retrieved_docs.extend(retrieve_license(vector_db, question, k=5))

    else:
        retrieved_docs.extend(retrieve_general(vector_db, question, k=12))

    unique_docs = get_unique_documents(retrieved_docs)
    keyword_docs = retrieve_keywords(vector_db, question, k=8)
    return fuse_rankings(unique_docs, keyword_docs, limit=20)
