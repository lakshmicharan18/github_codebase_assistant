from langchain_core.documents import Document


def get_unique_documents(documents):
    unique_docs = []
    seen = set()

    for doc in documents:
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        key = f"{source}-{chunk_id}"

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def multi_retrieve(vector_db, question, category):
    retrieved_docs = []

    if category == "architecture":
        repo_structure_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,
                "filter": {"file_type": "repo_structure"}
            }
        )

        readme_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 15,
                "filter": {"file_type": "readme"}
            }
        )

        docs_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 15,
                "filter": {"file_type": "documentation"}
            }
        )

        retrieved_docs.extend(repo_structure_retriever.invoke(question))
        retrieved_docs.extend(readme_retriever.invoke(question))
        retrieved_docs.extend(docs_retriever.invoke(question))

    elif category == "overview":
        readme_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 20,
                "filter": {"file_type": "readme"}
            }
        )

        docs_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 15,
                "filter": {"file_type": "documentation"}
            }
        )

        retrieved_docs.extend(readme_retriever.invoke(question))
        retrieved_docs.extend(docs_retriever.invoke(question))

    elif category == "implementation":
        source_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30,
                "filter": {"file_type": "source_code"}
            }
        )

        docs_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 15,
                "filter": {"file_type": "documentation"}
            }
        )

        retrieved_docs.extend(source_retriever.invoke(question))
        retrieved_docs.extend(docs_retriever.invoke(question))

    elif category == "testing":
        test_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30,
                "filter": {"file_type": "test"}
            }
        )

        retrieved_docs.extend(test_retriever.invoke(question))

    elif category == "configuration":
        config_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30,
                "filter": {"file_type": "configuration"}
            }
        )

        retrieved_docs.extend(config_retriever.invoke(question))

    elif category == "dependency":
        dependency_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30,
                "filter": {"file_type": "dependency"}
            }
        )

        retrieved_docs.extend(dependency_retriever.invoke(question))

    elif category == "license":
        license_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "filter": {"file_type": "license"}
            }
        )

        retrieved_docs.extend(license_retriever.invoke(question))

    else:
        general_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 12,
                "fetch_k": 40
            }
        )

        retrieved_docs.extend(general_retriever.invoke(question))

    unique_docs = get_unique_documents(retrieved_docs)

    unique_docs = sorted(
        unique_docs,
        key=lambda doc: doc.metadata.get("file_priority", 10)
    )

    return unique_docs[:10]