"""
Enhanced Bias Classifier with Advanced Text Analysis
Combines ML model with nuanced text analysis for better bias detection
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import sys

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)

from src.core.scrapper import extract_text_from_url
from src.utils.bias_utils import get_source_bias_features, extract_domain
from src.utils.advanced_text_analysis import analyze_text_for_bias

class EnhancedBiasClassifier:
    """
    Enhanced bias classifier that combines ML predictions with advanced text analysis.
    """
    
    def __init__(self, model_path: str = "models/simple_model.pkl"):
        """Initialize the enhanced classifier."""
        self.model_path = model_path
        self.model = None
        self.text_vectorizer = None
        self.bias_scaler = None
        self.bias_info = None
        
        # Load the trained model and components
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and components."""
        try:
            # Load the main model
            self.model = joblib.load(self.model_path)
            
            # Load the text vectorizer
            vectorizer_path = self.model_path.replace('simple_model.pkl', 'text_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.text_vectorizer = joblib.load(vectorizer_path)
            
            # Load the bias scaler
            scaler_path = self.model_path.replace('simple_model.pkl', 'bias_scaler.pkl')
            if os.path.exists(scaler_path):
                self.bias_scaler = joblib.load(scaler_path)
            
            # Load bias feature info
            bias_info_path = self.model_path.replace('simple_model.pkl', 'bias_info.pkl')
            if os.path.exists(bias_info_path):
                self.bias_info = joblib.load(bias_info_path)
                
            print(f"Enhanced model loaded from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the model first using train_model_simple.py")
    
    def predict_bias_enhanced(self, text: str, source_url: Optional[str] = None) -> Dict:
        """
        Enhanced bias prediction combining ML model with advanced text analysis.
        
        Args:
            text: The text to analyze
            source_url: Optional source URL for bias lookup
            
        Returns:
            Dictionary with comprehensive bias analysis
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        # Get basic ML prediction
        basic_result = self._get_basic_prediction(text, source_url)
        
        # Get advanced text analysis
        advanced_analysis = analyze_text_for_bias(text)
        
        # Combine and enhance the results
        enhanced_result = self._combine_analyses(basic_result, advanced_analysis, text, source_url)
        
        return enhanced_result
    
    def _get_basic_prediction(self, text: str, source_url: Optional[str] = None) -> Dict:
        """Get the basic ML model prediction."""
        try:
            # Prepare text features
            if self.text_vectorizer:
                text_features = self.text_vectorizer.transform([text]).toarray()
            else:
                text_features = np.zeros((1, 1000))  # Default size
            
            # Prepare bias features
            bias_features = np.zeros((1, 13))  # Default bias feature size
            source_bias_info = {}
            
            if source_url:
                source_features = get_source_bias_features(source_url)
                if source_features:
                    # Extract numerical features in the correct order
                    bias_feature_names = [
                        'allsides_score_normalized', 'adfontes_bias_score', 
                        'adfontes_reliability_score', 'score_difference', 'reliability_score'
                    ]
                    
                    for i, feature_name in enumerate(bias_feature_names):
                        if feature_name in source_features:
                            bias_features[0, i] = source_features[feature_name]
                    
                    source_bias_info = source_features
            
            # Scale bias features if scaler is available
            if self.bias_scaler:
                bias_features_scaled = self.bias_scaler.transform(bias_features)
            else:
                bias_features_scaled = bias_features
            
            # Combine features
            combined_features = np.hstack([text_features, bias_features_scaled])
            
            # Make prediction
            prediction = self.model.predict(combined_features)[0]
            probabilities = self.model.predict_proba(combined_features)[0]
            
            # Get confidence
            confidence = np.max(probabilities) * 100
            
            # Get confidence by class
            classes = self.model.classes_
            confidence_by_class = {class_name: prob * 100 for class_name, prob in zip(classes, probabilities)}
            
            return {
                'bias_label': prediction,
                'confidence': confidence,
                'confidence_by_class': confidence_by_class,
                'source_bias_info': source_bias_info,
                'text_length': len(text)
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {e}"}
    
    def _combine_analyses(self, basic_result: Dict, advanced_analysis: Dict, 
                         text: str, source_url: Optional[str] = None) -> Dict:
        """Combine basic ML prediction with advanced text analysis."""
        
        # Start with basic result
        result = basic_result.copy()
        
        # Add advanced analysis
        result['advanced_analysis'] = advanced_analysis
        
        # Enhance bias prediction based on advanced analysis
        enhanced_prediction = self._enhance_prediction_with_analysis(
            basic_result, advanced_analysis
        )
        
        result['enhanced_bias_label'] = enhanced_prediction['enhanced_label']
        result['enhanced_confidence'] = enhanced_prediction['enhanced_confidence']
        result['bias_indicators'] = enhanced_prediction['bias_indicators']
        result['nuance_score'] = enhanced_prediction['nuance_score']
        
        # Add source domain if available
        if source_url:
            result['source_domain'] = extract_domain(source_url)
        
        return result
    
    def _enhance_prediction_with_analysis(self, basic_result: Dict, 
                                        advanced_analysis: Dict) -> Dict:
        """Enhance the basic prediction using advanced text analysis."""
        
        # Extract key metrics from advanced analysis
        sentiment = advanced_analysis.get('sentiment', {})
        loaded_language = advanced_analysis.get('loaded_language', {})
        certainty_levels = advanced_analysis.get('certainty_levels', {})
        factual_claims = advanced_analysis.get('factual_claims', [])
        
        # Calculate bias indicators
        bias_indicators = {
            'emotional_intensity': self._calculate_emotional_intensity(sentiment),
            'loaded_language_score': self._calculate_loaded_language_score(loaded_language),
            'certainty_bias': self._calculate_certainty_bias(certainty_levels),
            'factual_claim_quality': self._assess_factual_claims(factual_claims),
            'overall_subjectivity': sentiment.get('subjectivity', 0.5)
        }
        
        # Calculate nuance score (how much the advanced analysis affects the prediction)
        nuance_score = self._calculate_nuance_score(bias_indicators)
        
        # Enhance the prediction
        enhanced_label = basic_result.get('bias_label', 'Unknown')
        enhanced_confidence = basic_result.get('confidence', 0)
        
        # Adjust confidence based on advanced analysis
        confidence_adjustment = self._calculate_confidence_adjustment(bias_indicators)
        enhanced_confidence = max(0, min(100, enhanced_confidence + confidence_adjustment))
        
        # Potentially adjust label based on strong indicators
        label_adjustment = self._suggest_label_adjustment(basic_result, bias_indicators)
        if label_adjustment:
            enhanced_label = label_adjustment
        
        return {
            'enhanced_label': enhanced_label,
            'enhanced_confidence': enhanced_confidence,
            'bias_indicators': bias_indicators,
            'nuance_score': nuance_score
        }
    
    def _calculate_emotional_intensity(self, sentiment: Dict) -> float:
        """Calculate emotional intensity score."""
        emotion_scores = sentiment.get('emotion_scores', {})
        
        # Weight different emotions
        emotion_weights = {
            'anger': 0.3,
            'fear': 0.2,
            'disgust': 0.3,
            'surprise': 0.1,
            'trust': -0.1,  # Negative weight for trust (reduces bias)
            'distrust': 0.2
        }
        
        total_intensity = 0
        for emotion, weight in emotion_weights.items():
            count = emotion_scores.get(emotion, 0)
            total_intensity += count * weight
        
        # Normalize to 0-1 scale
        return min(1.0, max(0.0, total_intensity / 10))
    
    def _calculate_loaded_language_score(self, loaded_language: Dict) -> float:
        """Calculate loaded language score."""
        total_score = 0
        
        # Weight different types of loaded language
        language_weights = {
            'exaggeration': 0.3,
            'minimization': 0.2,
            'qualifiers': 0.1,
            'intensifiers': 0.2,
            'dismissive': 0.4,
            'authoritative': 0.2,
            'hedging': -0.1  # Negative weight (reduces bias)
        }
        
        for category, weight in language_weights.items():
            category_data = loaded_language.get(category, {})
            density = category_data.get('density', 0)
            total_score += density * weight
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_certainty_bias(self, certainty_levels: Dict) -> float:
        """Calculate bias based on certainty levels."""
        high_certainty = certainty_levels.get('high_certainty', 0)
        low_certainty = certainty_levels.get('low_certainty', 0)
        uncertainty = certainty_levels.get('uncertainty', 0)
        
        # High certainty with low uncertainty suggests potential bias
        certainty_score = (high_certainty - low_certainty - uncertainty) / 10
        return min(1.0, max(0.0, (certainty_score + 1) / 2))
    
    def _assess_factual_claims(self, factual_claims: List[Dict]) -> Dict:
        """Assess the quality of factual claims."""
        if not factual_claims:
            return {'average_confidence': 0.5, 'average_verifiability': 'medium', 'claim_count': 0}
        
        confidences = [claim.get('confidence', 0.5) for claim in factual_claims]
        verifiability_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for claim in factual_claims:
            verifiability = claim.get('verifiability', 'medium')
            verifiability_counts[verifiability] += 1
        
        return {
            'average_confidence': np.mean(confidences),
            'average_verifiability': max(verifiability_counts, key=verifiability_counts.get),
            'claim_count': len(factual_claims),
            'verifiability_distribution': verifiability_counts
        }
    
    def _calculate_nuance_score(self, bias_indicators: Dict) -> float:
        """Calculate how much nuance the advanced analysis detected."""
        # Higher scores indicate more nuanced analysis
        factors = [
            bias_indicators['emotional_intensity'],
            bias_indicators['loaded_language_score'],
            bias_indicators['certainty_bias'],
            1 - bias_indicators['overall_subjectivity']  # More objective = more nuanced
        ]
        
        return np.mean(factors)
    
    def _calculate_confidence_adjustment(self, bias_indicators: Dict) -> float:
        """Calculate confidence adjustment based on advanced analysis."""
        # Strong indicators should increase confidence
        adjustment = 0
        
        # Emotional intensity affects confidence
        if bias_indicators['emotional_intensity'] > 0.5:
            adjustment += 5
        
        # Loaded language affects confidence
        if bias_indicators['loaded_language_score'] > 0.3:
            adjustment += 3
        
        # Certainty bias affects confidence
        if bias_indicators['certainty_bias'] > 0.6:
            adjustment += 2
        
        # High subjectivity reduces confidence
        if bias_indicators['overall_subjectivity'] > 0.7:
            adjustment -= 5
        
        return adjustment
    
    def _suggest_label_adjustment(self, basic_result: Dict, bias_indicators: Dict) -> Optional[str]:
        """Suggest label adjustments based on strong indicators."""
        current_label = basic_result.get('bias_label', '')
        confidence_by_class = basic_result.get('confidence_by_class', {})
        
        # If emotional intensity is very high, consider moving to extreme category
        if bias_indicators['emotional_intensity'] > 0.7:
            if current_label == 'Lean Left':
                return 'Far Left'
            elif current_label == 'Lean Right':
                return 'Far Right'
        
        # If loaded language is very high, consider moving to extreme category
        if bias_indicators['loaded_language_score'] > 0.6:
            if current_label == 'Lean Left':
                return 'Far Left'
            elif current_label == 'Lean Right':
                return 'Far Right'
        
        # If certainty bias is very high, consider moving to extreme category
        if bias_indicators['certainty_bias'] > 0.8:
            if current_label == 'Lean Left':
                return 'Far Left'
            elif current_label == 'Lean Right':
                return 'Far Right'
        
        return None  # No adjustment needed
    
    def predict_bias_from_url_enhanced(self, url: str) -> Dict:
        """
        Enhanced bias prediction from URL.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with comprehensive bias analysis
        """
        try:
            # Extract text from URL
            text = extract_text_from_url(url)
            if not text:
                return {"error": "Could not extract text from URL"}
            
            # Analyze with enhanced classifier
            return self.predict_bias_enhanced(text, url)
            
        except Exception as e:
            return {"error": f"URL analysis error: {e}"}

def main():
    """Test the enhanced classifier."""
    classifier = EnhancedBiasClassifier()
    
    # Test with the Post Millennial article
    test_text = """Former Biden advisor Anita Dunn claims he was 'appropriately accessible to the press' during presidency. 
    Biden was notoriously not available to the press during his term in office. He held the fewest press conferences of any president since the 1980s."""
    
    result = classifier.predict_bias_enhanced(test_text, "https://thepostmillennial.com/article")
    
    print("=== Enhanced Bias Analysis Results ===")
    print(f"Basic Prediction: {result.get('bias_label', 'Unknown')}")
    print(f"Enhanced Prediction: {result.get('enhanced_bias_label', 'Unknown')}")
    print(f"Confidence: {result.get('enhanced_confidence', 0):.1f}%")
    print(f"Nuance Score: {result.get('nuance_score', 0):.2f}")
    
    if 'bias_indicators' in result:
        indicators = result['bias_indicators']
        print(f"\nBias Indicators:")
        print(f"  Emotional Intensity: {indicators.get('emotional_intensity', 0):.2f}")
        print(f"  Loaded Language: {indicators.get('loaded_language_score', 0):.2f}")
        print(f"  Certainty Bias: {indicators.get('certainty_bias', 0):.2f}")
        print(f"  Subjectivity: {indicators.get('overall_subjectivity', 0):.2f}")

if __name__ == "__main__":
    main() 