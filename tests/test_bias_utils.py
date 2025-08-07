import unittest
from unittest.mock import patch
import sys
import os

# Add parent directory to path to import bias_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.bias_utils import (
    extract_domain, 
    normalize_allsides_to_adfontes, 
    normalize_adfontes_to_allsides,
    get_bias_info,
    get_source_bias_features,
    _get_consensus_bias
)

class TestBiasUtils(unittest.TestCase):
    
    def test_extract_domain(self):
        """Test domain extraction from various URL formats."""
        test_cases = [
            ("https://www.foxnews.com/politics", "foxnews.com"),
            ("http://nytimes.com/section/opinion", "nytimes.com"),
            ("https://www.bbc.com/news", "bbc.com"),
            ("reuters.com", "reuters.com"),
            ("www.cnn.com", "cnn.com"),
            ("https://www.washingtonpost.com/politics/2024/", "washingtonpost.com"),
            ("", None),
            (None, None),
        ]
        
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = extract_domain(url)
                self.assertEqual(result, expected)
    
    def test_normalize_allsides_to_adfontes(self):
        """Test AllSides to Ad Fontes score normalization."""
        test_cases = [
            (-6, -42),
            (-3, -21),
            (0, 0),
            (3, 21),
            (6, 42),
            (-1.42, -9.94),
            (2.74, 19.18),
        ]
        
        for allsides_score, expected in test_cases:
            with self.subTest(allsides_score=allsides_score):
                result = normalize_allsides_to_adfontes(allsides_score)
                self.assertAlmostEqual(result, expected, places=2)
    
    def test_normalize_adfontes_to_allsides(self):
        """Test Ad Fontes to AllSides score normalization."""
        test_cases = [
            (-42, -6),
            (-21, -3),
            (0, 0),
            (21, 3),
            (42, 6),
            (-8.06, -1.15),
            (11.06, 1.58),
        ]
        
        for adfontes_score, expected in test_cases:
            with self.subTest(adfontes_score=adfontes_score):
                result = normalize_adfontes_to_allsides(adfontes_score)
                self.assertAlmostEqual(result, expected, places=2)
    
    def test_get_bias_info_existing_domain(self):
        """Test bias info retrieval for domains in both dictionaries."""
        # Test with foxnews.com which should be in both
        result = get_bias_info("foxnews.com")
        
        self.assertIsNotNone(result)
        self.assertEqual(result['domain'], "foxnews.com")
        self.assertIsNotNone(result['allsides'])
        self.assertIsNotNone(result['adfontes'])
        self.assertIsNotNone(result['combined'])
        
        # Check combined analysis
        combined = result['combined']
        self.assertIn('allsides_normalized', combined)
        self.assertIn('adfontes_score', combined)
        self.assertIn('score_difference', combined)
        self.assertIn('agreement_level', combined)
        self.assertIn('reliability_score', combined)
        self.assertIn('consensus_bias', combined)
    
    def test_get_bias_info_nonexistent_domain(self):
        """Test bias info retrieval for domain not in dictionaries."""
        result = get_bias_info("nonexistentdomain.com")
        
        self.assertIsNotNone(result)
        self.assertEqual(result['domain'], "nonexistentdomain.com")
        self.assertIsNone(result['allsides'])
        self.assertIsNone(result['adfontes'])
        self.assertEqual(result['combined'], {})
    
    def test_get_source_bias_features(self):
        """Test feature generation for ML model."""
        features = get_source_bias_features("foxnews.com")
        
        self.assertIsNotNone(features)
        self.assertEqual(features['domain'], "foxnews.com")
        self.assertTrue(features['has_allsides'])
        self.assertTrue(features['has_adfontes'])
        
        # Check AllSides features
        self.assertIn('allsides_label', features)
        self.assertIn('allsides_score', features)
        self.assertIn('allsides_score_normalized', features)
        
        # Check Ad Fontes features
        self.assertIn('adfontes_label', features)
        self.assertIn('adfontes_bias_score', features)
        self.assertIn('adfontes_reliability_score', features)
        
        # Check combined features
        self.assertIn('score_difference', features)
        self.assertIn('agreement_level', features)
        self.assertIn('consensus_bias', features)
        self.assertIn('reliability_score', features)
    
    def test_get_consensus_bias(self):
        """Test consensus bias determination."""
        test_cases = [
            # Agreement cases
            (("Right", "Skews Right"), "Right"),
            (("Left", "Strong Left"), "Left"),
            (("Center", "Middle or Balanced Bias"), "Center"),
            
            # Disagreement cases
            (("Right", "Skews Left"), "Mixed (Right/Left)"),
            (("Center", "Strong Right"), "Mixed (Center/Right)"),
            (("Left", "Middle or Balanced Bias"), "Mixed (Left/Center)"),
            
            # Single source cases
            (("Right", None), "Right"),
            ((None, "Skews Left"), "Left"),
            ((None, None), None),
        ]
        
        for (allsides_label, adfontes_label), expected in test_cases:
            with self.subTest(allsides=allsides_label, adfontes=adfontes_label):
                result = _get_consensus_bias(allsides_label, adfontes_label)
                self.assertEqual(result, expected)
    
    def test_agreement_levels(self):
        """Test agreement level calculation based on score differences."""
        # Test with foxnews.com to check agreement level
        result = get_bias_info("foxnews.com")
        combined = result['combined']
        
        # Check that agreement level is one of the expected values
        self.assertIn(combined['agreement_level'], ["High", "Medium", "Low"])
        
        # Check that score difference is calculated correctly
        self.assertGreaterEqual(combined['score_difference'], 0)
    
    def test_feature_consistency(self):
        """Test that features are consistent between get_bias_info and get_source_bias_features."""
        domain = "foxnews.com"
        bias_info = get_bias_info(domain)
        features = get_source_bias_features(domain)
        
        # Check that normalized scores match
        if bias_info['combined']:
            expected_normalized = bias_info['combined']['allsides_normalized']
            actual_normalized = features['allsides_score_normalized']
            self.assertAlmostEqual(expected_normalized, actual_normalized, places=2)

if __name__ == '__main__':
    unittest.main() 