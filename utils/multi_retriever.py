from langchain_core.documents import Document


def get_unique_documents(documents):
    unique_docs = []
    seen = set()

    for doc in documents:
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        content_preview = doc.page_content[:100]

        key = f"{source}-{chunk_id}-{content_preview}"

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

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

    if category == "architecture":
        retrieved_docs.extend(retrieve_repo_structure(vector_db, question, k=3))
        retrieved_docs.extend(retrieve_readme_first(vector_db, question, k=4))
        retrieved_docs.extend(retrieve_documentation(vector_db, question, k=4))

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
    sorted_docs = sort_documents_by_priority(unique_docs)

    return sorted_docs[:20]