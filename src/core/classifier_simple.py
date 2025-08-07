import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import bias_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..utils.bias_utils import get_source_bias_features, extract_domain

class SimpleBiasClassifier:
    """Simple bias classifier that uses both text and source bias features."""
    
    def __init__(self, model_path="models/simple_model.pkl"):
        """Initialize the classifier with trained model."""
        try:
            self.model = joblib.load(model_path)
            self.text_vectorizer = joblib.load("models/text_vectorizer.pkl")
            self.bias_scaler = joblib.load("models/bias_scaler.pkl")
            self.bias_info = joblib.load("models/bias_info.pkl")
            print(f"Loaded simple model from {model_path}")
        except FileNotFoundError as e:
            print(f"Model not found: {e}")
            print("Please run the training script first: python training/train_model_simple.py")
            self.model = None
    
    def predict_bias(self, text, source_url=None):
        """
        Predict bias for given text and optional source URL.
        
        Args:
            text (str): The article text to analyze
            source_url (str, optional): The source URL for bias lookup
            
        Returns:
            dict: Prediction results including bias label, confidence, and metadata
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Extract bias features from source URL
            if source_url:
                bias_features = get_source_bias_features(source_url)
                if not bias_features:
                    # Create default features for unknown sources
                    bias_features = {
                        'domain': extract_domain(source_url),
                        'has_allsides': False,
                        'has_adfontes': False,
                        'allsides_label': 'Unknown',
                        'allsides_score': 0.0,
                        'allsides_score_normalized': 0.0,
                        'adfontes_label': 'Unknown',
                        'adfontes_bias_score': 0.0,
                        'adfontes_reliability_score': 30.0,
                        'score_difference': 0.0,
                        'agreement_level': 'Unknown',
                        'consensus_bias': 'Unknown',
                        'reliability_score': 30.0
                    }
            else:
                # No source URL provided
                bias_features = {
                    'domain': 'unknown',
                    'has_allsides': False,
                    'has_adfontes': False,
                    'allsides_label': 'Unknown',
                    'allsides_score': 0.0,
                    'allsides_score_normalized': 0.0,
                    'adfontes_label': 'Unknown',
                    'adfontes_bias_score': 0.0,
                    'adfontes_reliability_score': 30.0,
                    'score_difference': 0.0,
                    'agreement_level': 'Unknown',
                    'consensus_bias': 'Unknown',
                    'reliability_score': 30.0
                }
            
            # Prepare text features
            X_text = self.text_vectorizer.transform([text])
            
            # Prepare bias features
            bias_numeric = pd.DataFrame([{
                'allsides_score_normalized': bias_features['allsides_score_normalized'],
                'adfontes_bias_score': bias_features['adfontes_bias_score'],
                'reliability_score': bias_features['reliability_score']
            }]).fillna(0)
            
            X_bias = self.bias_scaler.transform(bias_numeric)
            
            # Combine features
            X_combined = np.hstack([X_text.toarray(), X_bias])
            
            # Make prediction
            prediction = self.model.predict(X_combined)[0]
            probabilities = self.model.predict_proba(X_combined)[0]
            confidence = max(probabilities)
            
            # Get class labels
            class_labels = self.model.classes_
            confidence_by_class = dict(zip(class_labels, probabilities))
            
            return {
                'bias_label': prediction,
                'confidence': round(confidence, 3),
                'confidence_by_class': {k: round(v, 3) for k, v in confidence_by_class.items()},
                'source_bias_info': bias_features,
                'text_length': len(text),
                'has_source_bias': bias_features['has_allsides'] or bias_features['has_adfontes']
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_bias_from_url(self, url):
        """
        Predict bias for content from a URL (combines scraping and bias prediction).
        
        Args:
            url (str): The URL to scrape and analyze
            
        Returns:
            dict: Combined scraping and bias prediction results
        """
        try:
            from .scrapper import extract_text_from_url
            
            # Extract text from URL
            text = extract_text_from_url(url)
            if not text:
                return {"error": "Failed to extract text from URL"}
            
            # Predict bias using the extracted text and source URL
            result = self.predict_bias(text, url)
            result['extracted_text'] = text[:500] + "..." if len(text) > 500 else text
            result['source_url'] = url
            
            return result
            
        except Exception as e:
            return {"error": f"URL analysis failed: {str(e)}"}

# Convenience function for backward compatibility
def predict_bias(text, source_url=None):
    """Simple function to predict bias for text."""
    classifier = SimpleBiasClassifier()
    result = classifier.predict_bias(text, source_url)
    return result.get('bias_label', 'Error')

# Example usage
if __name__ == "__main__":
    # Test the simple classifier
    classifier = SimpleBiasClassifier()
    
    # Test with sample text and source URL
    sample_text = "Government announces new climate policy to reduce emissions by 50% by 2030."
    sample_url = "https://www.nytimes.com/article1"
    
    print("=== Simple Bias Classifier Test ===")
    print(f"Text: {sample_text}")
    print(f"Source: {sample_url}")
    
    result = classifier.predict_bias(sample_text, sample_url)
    
    if 'error' not in result:
        print(f"\nPrediction: {result['bias_label']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Source Bias: {result['source_bias_info']['consensus_bias']}")
        print(f"Reliability Score: {result['source_bias_info']['reliability_score']}")
        print(f"Confidence by Class: {result['confidence_by_class']}")
    else:
        print(f"Error: {result['error']}") 