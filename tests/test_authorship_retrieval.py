import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from utils.query_router import route_question
from utils.query_intent import is_authorship_question
from utils.multi_retriever import multi_retrieve
from utils.reranker import DocumentReranker


class AuthorshipTests(unittest.TestCase):
    def test_route_without_model_and_leave_implementation_alone(self):
        with patch('utils.query_router.ChatGroq', side_effect=AssertionError('Unexpected API call')):
            self.assertEqual(route_question('who built this project', None), 'overview')
            self.assertEqual(route_question('who built project', None), 'overview')
        self.assertFalse(is_authorship_question('Who created the database connection?'))
        self.assertFalse(is_authorship_question('How does the project build its index?'))

    def test_author_evidence_survives_low_reranker_score(self):
        author = Document(page_content='Developed by Example Developer.', metadata={'source':'README.md','file_type':'readme'})
        noise = Document(page_content='Project installation instructions', metadata={'source':'README.md','file_type':'readme'})
        with patch('utils.multi_retriever.retrieve_by_filename_mention', return_value=[]), \
             patch('utils.multi_retriever.retrieve_readme_first', return_value=[noise, author]) as readme, \
             patch('utils.multi_retriever.retrieve_documentation', return_value=[]), \
             patch('utils.multi_retriever.retrieve_keywords', return_value=[author]):
            candidates = multi_retrieve(object(), 'Who built this project?', 'general')
            self.assertIn('developed by', readme.call_args.args[1])
        class Scores:
            def predict(self, pairs):
                return [-10 if 'Developed by' in text else 10 for _, text in pairs]
        reranker = DocumentReranker.__new__(DocumentReranker)
        reranker.model = Scores()
        result = reranker.rerank('Who built this project?', candidates, top_k=1)
        self.assertEqual(result[0].page_content, author.page_content)
