"""In-memory BM25 keyword index, owned by one repository's vector store."""
import math
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document


def tokenize(text):
    """Keep full identifiers and also match snake_case and camelCase parts."""
    tokens = []
    for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+", text):
        whole = word.lower()
        tokens.append(whole)
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").lower().split()
        tokens.extend(part for part in parts if part != whole)
    return tokens


class BM25Index:
    def __init__(self, documents):
        self.documents = list(documents)
        self.postings = defaultdict(list)
        self.lengths = []
        for i, doc in enumerate(self.documents):
            terms = tokenize(doc.metadata.get("source", "") + "\n" + doc.page_content)
            self.lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((i, frequency))
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1) or 1

    def retrieve(self, question, k=8, file_types=None):
        scores = defaultdict(float)
        count = len(self.documents)
        for term in set(tokenize(question)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, frequency in postings:
                if file_types is not None and self.documents[i].metadata.get("file_type") not in file_types:
                    continue
                normalization = 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average_length)
                scores[i] += idf * frequency * 2.5 / (frequency + normalization)
        return [
            Document(page_content=self.documents[i].page_content,
                     metadata=dict(self.documents[i].metadata))
            for i in sorted(scores, key=lambda i: (-scores[i], i))[:max(k, 0)]
        ]


def attach_bm25_index(vector_db, documents=None):
    if documents is None:
        stored = vector_db.get(include=["documents", "metadatas"])
        documents = [Document(page_content=text, metadata=meta or {})
                     for text, meta in zip(stored["documents"], stored["metadatas"])]
    vector_db._bm25_index = BM25Index(documents)
    return vector_db._bm25_index


def retrieve_keywords(vector_db, question, k=8, file_types=None):
    index = getattr(vector_db, "_bm25_index", None)
    if index is None:
        index = attach_bm25_index(vector_db)
    return index.retrieve(question, k=k, file_types=file_types)


def fuse_rankings(*rankings, limit=20):
    """Reciprocal rank fusion: combine ranks, not incompatible raw scores."""
    scores, documents = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, start=1):
            key = (doc.metadata.get("source", ""), doc.page_content)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            if key not in documents:
                documents[key] = doc
            elif doc.metadata.get("filename_match"):
                documents[key].metadata["filename_match"] = True
    ordered = sorted(scores, key=lambda key: -scores[key])
    # Preserve explicitly named file evidence through the candidate budget.
    named = [key for key in ordered if documents[key].metadata.get("filename_match")][:limit]
    selected = set(named)
    keys = named + [key for key in ordered if key not in selected][:max(0, limit-len(named))]
    return [documents[key] for key in keys]


def retrieve_readme_introduction(vector_db):
    index = getattr(vector_db, "_bm25_index", None)
    if index is None:
        index = attach_bm25_index(vector_db)
    candidates = [doc for doc in index.documents
                  if doc.metadata.get("file_type") == "readme"
                  and doc.metadata.get("chunk_id") == 0]
    candidates.sort(key=lambda doc: (doc.metadata.get("source", "").count("/"),
                                     doc.metadata.get("source", "")))
    return [Document(page_content=doc.page_content,
                     metadata={**doc.metadata, "overview_evidence": True})
            for doc in candidates[:1]]
