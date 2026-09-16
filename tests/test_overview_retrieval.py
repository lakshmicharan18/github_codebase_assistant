import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from utils.bm25_retriever import attach_bm25_index, retrieve_keywords
from utils.multi_retriever import multi_retrieve
from utils.reranker import DocumentReranker
from utils.query_router import route_question


class OverviewTests(unittest.TestCase):
    def test_purpose_preserves_intro_and_excludes_keyword_code_noise(self):
        intro = Document(page_content='A tool for understanding unfamiliar repositories.', metadata={'source':'README.md','file_type':'readme','chunk_id':0})
        example = Document(page_content='What problem does this project solve?', metadata={'source':'README.md','file_type':'readme','chunk_id':1})
        noise = Document(page_content='What problem does this project solve?', metadata={'source':'tests/test.py','file_type':'test','chunk_id':0})
        class Store: pass
        store = Store()
        attach_bm25_index(store, [intro, example, noise])
        with patch('utils.query_router.ChatGroq', side_effect=AssertionError('API call')):
            self.assertEqual(route_question('What problem does this project solve?', None), 'overview')
        with patch('utils.multi_retriever.retrieve_by_filename_mention', return_value=[]), patch('utils.multi_retriever.retrieve_readme_first', return_value=[example]), patch('utils.multi_retriever.retrieve_documentation', return_value=[]):
            docs = multi_retrieve(store, 'What problem does this project solve?', 'general')
        self.assertFalse(any(d.metadata['file_type']=='test' for d in docs))
        self.assertEqual(docs[0].page_content, intro.page_content)
        class Scores:
            def predict(self, pairs): return [-10 if text==intro.page_content else 10 for _,text in pairs]
        reranker=DocumentReranker.__new__(DocumentReranker)
        reranker.model=Scores()
        self.assertEqual(reranker.rerank('purpose', docs, top_k=1)[0].page_content, intro.page_content)
        self.assertTrue(any(d.metadata['file_type']=='test' for d in retrieve_keywords(store, 'problem')))
