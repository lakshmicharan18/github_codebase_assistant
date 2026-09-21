import ast
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_python_document(doc, fallback):
    """Respect Python statement boundaries even when the loader numbered lines.

    Keep functions and prompt assignments together if they fit the existing
    1,800-character budget. Oversized statements use bounded recursive splits.
    """
    lines = doc.page_content.splitlines(keepends=True)
    raw = ''.join(re.sub(r"^\d+: ", "", line) for line in lines)
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return fallback.split_documents([doc])
    boundaries = {0, len(lines)}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and hasattr(node, 'end_lineno'):
            start = node.lineno - 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([start] + [d.lineno - 1 for d in node.decorator_list])
            # Don't break inside a statement that fits the chunk budget.
            if len(''.join(lines[start:node.end_lineno])) <= 1800:
                boundaries.add(start)
                boundaries.add(node.end_lineno)
    # Prefer the largest enclosing statement that fits; remove its inner cuts.
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and hasattr(node, 'end_lineno'):
            start = min([node.lineno - 1] + [d.lineno - 1 for d in getattr(node, 'decorator_list', [])])
            end = node.end_lineno
            if len(''.join(lines[start:end])) <= 1800:
                spans.append((start, end))
    for start, end in spans:
        boundaries.difference_update(b for b in list(boundaries) if start < b < end)
    cuts = sorted(boundaries)
    chunks, pending = [], ''
    for start, end in zip(cuts, cuts[1:]):
        section = ''.join(lines[start:end])
        if pending and len(pending + section) > 1800:
            chunks.extend(fallback.split_documents([Document(page_content=pending, metadata=dict(doc.metadata))]))
            pending = ''
        pending += section
    if pending.strip():
        chunks.extend(fallback.split_documents([Document(page_content=pending, metadata=dict(doc.metadata))]))
    return chunks


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

        elif file_extension == ".py":
            chunks = split_python_document(doc, code_splitter)

        else:
            chunks = code_splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_source"] = chunk.metadata.get("source")

        all_chunks.extend(chunks)

    return all_chunks
