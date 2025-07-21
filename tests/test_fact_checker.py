import unittest
from unittest.mock import patch, Mock
import os

class TestQueryGoogleFactCheck(unittest.TestCase):
    @patch('fact_checker.requests.get')
    def test_query_google_fact_check_success(self, mock_get):
        # Mock a successful API response
        mock_response = Mock()
        mock_response.json.return_value = {'claims': [{'text': 'Test claim', 'claimReview': []}]}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        from fact_checker import query_google_fact_check
        result = query_google_fact_check('Test claim')
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]['text'], 'Test claim')

    @patch('fact_checker.requests.get')
    def test_query_google_fact_check_api_error(self, mock_get):
        # Mock an API error response (no 'claims' key)
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {'error': 'Forbidden'}
        mock_get.return_value = mock_response

        from fact_checker import query_google_fact_check
        result = query_google_fact_check('Test claim')
        self.assertEqual(result, [])

    @patch('fact_checker.os.getenv', return_value=None)
    def test_missing_api_key_raises(self, mock_getenv):
        # Patch os.getenv to simulate missing API key
        with self.assertRaises(ValueError):
            import importlib
            import fact_checker
            importlib.reload(fact_checker)

if __name__ == '__main__':
    unittest.main() 