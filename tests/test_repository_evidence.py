import unittest
from pathlib import Path
from langchain_core.documents import Document
from utils.repo_loader import add_line_numbers
from utils.code_splitter import split_code_files
from utils.bm25_retriever import attach_bm25_index
from utils.evidence_retriever import retrieve_implementation_evidence


class RepositoryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Store: pass
        cls.store = Store()
        root=Path(__file__).resolve().parents[1]
        files=list((root/'utils').glob('*.py'))+[root/'app.py']
        docs=[Document(page_content=add_line_numbers(p.read_text()), metadata={
            'source':str(p.relative_to(root)), 'file_type':'source_code','file_extension':'.py'}) for p in files]
        cls.chunks=split_code_files(docs)
        attach_bm25_index(cls.store, cls.chunks)

    def test_model_configuration_evidence(self):
        docs=retrieve_implementation_evidence(self.store,'Which LLM generates answers and can it be configured?')
        self.assertTrue(any('GROQ_MODEL' in d.page_content for d in docs))

    def test_missing_context_evidence(self):
        docs=retrieve_implementation_evidence(self.store,'What happens when retrieved context does not contain the answer?')
        self.assertTrue(any("I don't know from this codebase" in d.page_content for d in docs))

    def test_personal_key_evidence(self):
        docs=retrieve_implementation_evidence(self.store,'Does Reset Session remove the personal API key?')
        self.assertTrue(any('personal_groq_key' in d.page_content for d in docs))

    def test_bounded_chunks_and_complete_numbered_function(self):
        text='@cache\ndef sample():\n    """Example."""\n    return 42\n'
        doc=Document(page_content=add_line_numbers(text),metadata={'file_extension':'.py'})
        chunks=split_code_files([doc])
        self.assertTrue(any('1: @cache' in c.page_content and '4:     return 42' in c.page_content for c in chunks))
        self.assertTrue(all(len(c.page_content)<=1800 for c in self.chunks))
