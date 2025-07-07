import unittest
from unittest.mock import patch, Mock
from scrapper import extract_text_from_url

class TestExtractTextFromUrl(unittest.TestCase):
    @patch('scrapper.requests.get')
    def test_extracts_paragraph_text(self, mock_get):
        html = '''<html><body><p>Hello</p><p>World</p></body></html>'''
        mock_get.return_value = Mock(text=html)
        result = extract_text_from_url('http://example.com')
        self.assertEqual(result, 'Hello\nWorld')

    @patch('scrapper.requests.get')
    def test_removes_unwanted_tags(self, mock_get):
        html = '''<html><body><header>Header</header><p>Keep me</p><footer>Footer</footer></body></html>'''
        mock_get.return_value = Mock(text=html)
        result = extract_text_from_url('http://example.com')
        self.assertEqual(result, 'Keep me')

    @patch('scrapper.requests.get')
    def test_handles_no_paragraphs(self, mock_get):
        html = '<html><body><div>No paragraphs here</div></body></html>'
        mock_get.return_value = Mock(text=html)
        result = extract_text_from_url('http://example.com')
        self.assertEqual(result, '')

    @patch('scrapper.requests.get')
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception('Network error')
        with self.assertRaises(Exception):
            extract_text_from_url('http://badurl.com')

if __name__ == '__main__':
    unittest.main() 