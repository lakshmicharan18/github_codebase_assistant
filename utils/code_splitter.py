from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_code_files(documents):
    all_chunks = []

    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=[
            # Split before decorators first so decorated definitions stay
            # together when they fit within the existing chunk budget.
            "\n@",
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n    @",
            "\n    def ",
            "\n    async def ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    markdown_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    structure_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )

    for doc in documents:
        file_type = doc.metadata.get("file_type")
        file_extension = doc.metadata.get("file_extension")

        if file_type == "repo_structure":
            chunks = structure_splitter.split_documents([doc])

        elif file_extension in [".md", ".rst", ".txt"]:
            chunks = markdown_splitter.split_documents([doc])

        else:
            chunks = code_splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_source"] = chunk.metadata.get("source")

        all_chunks.extend(chunks)

    return all_chunks
