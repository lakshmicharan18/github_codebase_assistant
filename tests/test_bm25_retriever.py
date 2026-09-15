import unittest
from langchain_core.documents import Document
from utils.bm25_retriever import BM25Index, fuse_rankings, retrieve_keywords


def doc(text, source='code.py', **metadata):
    return Document(page_content=text, metadata={'source': source, **metadata})


class KeywordTests(unittest.TestCase):
    def test_identifier_parts_and_exact_keyword(self):
        index = BM25Index([doc('def create_vector_store(): pass'), doc('def renderPage(): pass')])
        self.assertIn('create_vector_store', index.retrieve('vector store')[0].page_content)
        self.assertIn('renderPage', index.retrieve('renderPage')[0].page_content)
        self.assertEqual(index.retrieve('unmatched_token_xyz'), [])
        self.assertEqual(BM25Index([]).retrieve('anything'), [])

    def test_reload_is_cached_and_isolated(self):
        class Store:
            def __init__(self, text):
                self.text, self.calls = text, 0
            def get(self, **kwargs):
                self.calls += 1
                return {'documents': [self.text], 'metadatas': [{'source': 'code.py'}]}
        first, second = Store('alpha'), Store('beta')
        self.assertEqual(len(retrieve_keywords(first, 'alpha')), 1)
        self.assertEqual(retrieve_keywords(second, 'alpha'), [])
        result = retrieve_keywords(first, 'alpha')
        result[0].metadata['filename_match'] = True
        self.assertNotIn('filename_match', retrieve_keywords(first, 'alpha')[0].metadata)
        self.assertEqual(first.calls, 1)

    def test_fusion_combines_evidence_and_keeps_named_files(self):
        shared = doc('shared')
        vector = [doc('semantic'), shared]
        keyword = [doc('keyword'), doc('shared')]
        fused = fuse_rankings(vector, keyword)
        self.assertEqual(fused[0].page_content, 'shared')
        self.assertEqual(len(fused), 3)
        named = doc('explicit', filename_match=True)
        fused = fuse_rankings(vector + [named], keyword, limit=2)
        self.assertEqual(len(fused), 2)
        self.assertTrue(any(d.metadata.get('filename_match') for d in fused))


if __name__ == '__main__':
    unittest.main()
