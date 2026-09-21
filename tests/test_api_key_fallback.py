import os
import unittest
from unittest.mock import patch, MagicMock
import httpx
from groq import RateLimitError, AuthenticationError
from streamlit.testing.v1 import AppTest


def api_error(kind, status):
    response = httpx.Response(status, request=httpx.Request('POST', 'https://api.groq.com/openai/v1/chat/completions'))
    return kind('simulated error', response=response, body={})


class KeyFallbackTests(unittest.TestCase):
    def test_rate_limit_retry_and_reset(self):
        with patch.dict(os.environ, {'GROQ_API_KEY':'owner-test-key'}), \
             patch('utils.question_rewriter.rewrite_question', side_effect=lambda q,h,k:q) as rewrite, \
             patch('utils.rag_chain.create_rag_chain') as create, \
             patch('utils.vector_store.delete_vector_store'):
            chain=MagicMock()
            chain.invoke.side_effect=[api_error(RateLimitError,429), MagicMock(content='Recovered answer')]
            create.return_value=(chain,'overview',[],'context')
            app=AppTest.from_file('app.py', default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertFalse(any(w.label=='Your Groq API key' for w in app.text_input))
            app.session_state['vector_db']=object()
            app.chat_input[0].set_value('What is the project?').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['pending_question'],'What is the project?')
            key=next(w for w in app.text_input if w.label=='Your Groq API key')
            key.set_value('personal-test-key').run()
            next(b for b in app.button if b.label=='Retry previous question').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(rewrite.call_args.args[2],'personal-test-key')
            self.assertEqual(create.call_args.args[1],'personal-test-key')
            self.assertEqual(len(app.session_state['messages']),2)
            self.assertEqual(os.environ['GROQ_API_KEY'],'owner-test-key')
            next(b for b in app.button if b.label=='Reset Session (clear repo + chat)').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('personal_groq_key', app.session_state)

    def test_invalid_key_keeps_question_for_retry(self):
        with patch.dict(os.environ, {'GROQ_API_KEY':'owner-test-key'}), \
             patch('utils.question_rewriter.rewrite_question', side_effect=api_error(AuthenticationError,401)):
            app=AppTest.from_file('app.py', default_timeout=30).run()
            app.session_state['vector_db']=object()
            app.session_state['shared_key_limited']=True
            app.run()
            next(w for w in app.text_input if w.label=='Your Groq API key').set_value('bad-key').run()
            app.chat_input[0].set_value('Explain this project').run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state['pending_question'],'Explain this project')

    def test_shared_session_limit_enables_personal_key(self):
        import time
        from collections import deque
        with patch.dict(os.environ, {'GROQ_API_KEY':'owner-test-key'}), \
             patch('utils.question_rewriter.rewrite_question') as rewrite:
            app=AppTest.from_file('app.py', default_timeout=30).run()
            app.session_state['vector_db']=object()
            app.session_state['request_timestamps']=deque([time.time()]*20)
            app.chat_input[0].set_value('Another question').run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['shared_key_limited'])
            self.assertTrue(any(w.label=='Your Groq API key' for w in app.text_input))
            rewrite.assert_not_called()
