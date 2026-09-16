import tempfile
import unittest
from pathlib import Path
from langchain_core.documents import Document
from utils.repo_loader import load_code_files
from utils.reranker import DocumentReranker


class EvidenceTests(unittest.TestCase):
    def test_answer_keys_excluded_but_eval_code_kept(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'eval'
            folder.mkdir()
            (folder / 'golden_correctness_questions.py').write_text('SECRET_ANSWER_KEY')
            (folder / 'golden_router_questions.py').write_text('SECRET_ROUTER_KEY')
            (folder / 'evaluate_ragas.py').write_text('def evaluate(): pass')
            docs = load_code_files(root)
            content = '\n'.join(d.page_content for d in docs)
            self.assertNotIn('SECRET_', content)
            self.assertNotIn('golden_correctness_questions.py', content)
            self.assertTrue(any(d.metadata['source'] == 'eval/evaluate_ragas.py' for d in docs))

    def test_named_file_wins_over_high_scoring_commentary(self):
        class Scores:
            def predict(self, pairs):
                return [10, -5, -7, -8, -9]
        reranker = DocumentReranker.__new__(DocumentReranker)
        reranker.model = Scores()
        docs = [Document(page_content='commentary')] + [
            Document(page_content=str(i), metadata={'filename_match': True}) for i in range(4)
        ]
        selected = reranker.rerank('explain module.py', docs, top_k=3)
        self.assertEqual([d.page_content for d in selected], ['0', '1', '2'])
        selected = reranker.rerank('explain module.py', docs, top_k=6)
        self.assertEqual(selected[-1].page_content, 'commentary')
        self.assertEqual(reranker.rerank('anything', []), [])


if __name__ == '__main__':
    unittest.main()
