import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class DocumentReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, question, documents, top_k=6, guaranteed_filename_matches=6):
        """
        Reranks retrieved documents based on their relevance to the user's
        question. Chunks tagged filename_match (an explicitly named file,
        see multi_retriever.retrieve_by_filename_mention) are guaranteed up
        to `guaranteed_filename_matches` of the final top_k slots, even if
        their raw score wouldn't otherwise make the cut. Without this, a
        long file mentioned by name can lose most of its chunks to unrelated
        but higher-scoring chunks, leaving the answer citing the right file
        but missing most of its actual content.
        """

        if not documents:
            return []

        question_document_pairs = [
            [question, doc.page_content]
            for doc in documents
        ]

        scores = self.model.predict(question_document_pairs)

        scored_documents = []

        for doc, score in zip(documents, scores):
            doc.metadata["reranker_score"] = float(score)
            scored_documents.append(doc)

        scored_documents.sort(
            key=lambda doc: doc.metadata.get("reranker_score", 0),
            reverse=True
        )

        guaranteed_filename_matches = min(guaranteed_filename_matches, top_k)

        guaranteed = [
            doc for doc in scored_documents
            if doc.metadata.get("filename_match")
        ][:guaranteed_filename_matches]

        # Retain up to two documentation passages with explicit authorship evidence.
        author_docs = [doc for doc in scored_documents
                       if (doc.metadata.get("authorship_evidence") or doc.metadata.get("overview_evidence") or doc.metadata.get("implementation_evidence"))
                       and all(doc is not saved for saved in guaranteed)]
        guaranteed.extend(author_docs[:min(2, max(0, top_k - len(guaranteed)))])

        guaranteed_ids = {id(doc) for doc in guaranteed}
        remaining_slots = top_k - len(guaranteed)

        fill = [
            doc for doc in scored_documents
            if id(doc) not in guaranteed_ids
        ][:remaining_slots]

        result = guaranteed + fill
        # Put the explicitly requested file before supplemental evidence.
        # Each group is already sorted by cross-encoder relevance.

        return result


@st.cache_resource(show_spinner="Loading reranker model...")
def get_reranker():
    """
    Returns a cached DocumentReranker instance, shared across all sessions
    in this server process. Loading the cross-encoder is slow, so we want
    exactly one instance per process, not one per import or per session.
    """
    return DocumentReranker()