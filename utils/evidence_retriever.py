"""Find implementation evidence by query meaning, without repository-specific paths."""
import re
from utils.bm25_retriever import retrieve_keywords


def evidence_query(question):
    text = question.lower()
    if re.search(r'\b(?:llm|language model|embedding|embeddings)\b', text):
        if 'embedding' in text:
            return 'embedding model embeddings model_name HuggingFaceEmbeddings OpenAIEmbeddings', r'(?i)embedding|model_name'
        return 'model_name model getenv environ configuration ChatGroq', r'(?i)model_name\s*=|os\.getenv\('
    if re.search(r'\b(?:context|retriev\w*|evidence)\b', text) and re.search(r"missing|does not|doesn't|not contain|no answer|insufficient|cannot|can't", text):
        return "context answer missing evidence don't know unsupported prompt", r"(?i)don't know|do not know|unsupported|no retrieved evidence|not present"
    if 'reset session' in text or 'clear chat' in text:
        label = 'Reset Session' if 'reset session' in text else 'Clear Chat'
        return label + ' button session_state pop', r'(?i)button\([\"\']' + label
    if re.search(r'api.?key|personal key|rate.limit|clear chat|reset session|failed question', text):
        return question + ' session_state pending_question RateLimitError AuthenticationError api_key', r'(?i)except\s+(?:RateLimitError|AuthenticationError)|st\.session_state[.\[]|os\.getenv\('
    return None


def retrieve_implementation_evidence(vector_db, question):
    hint = evidence_query(question)
    if hint is None:
        return []
    expanded, pattern = hint
    matches = retrieve_keywords(vector_db, expanded, k=8, file_types={'source_code', 'configuration'})
    selected = []
    for doc in matches:
        if re.search(pattern, doc.page_content):
            doc.metadata['implementation_evidence'] = True
            selected.append(doc)
    return selected[:2]
