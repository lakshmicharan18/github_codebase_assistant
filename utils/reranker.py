from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class DocumentReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, question, documents, top_k=6):
        """
        Reranks retrieved documents based on their relevance
        to the user's question.
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

        return scored_documents[:top_k]