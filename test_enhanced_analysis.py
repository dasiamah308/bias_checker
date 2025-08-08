#!/usr/bin/env python3
"""
Test script to demonstrate enhanced bias analysis capabilities.
Shows the difference between basic ML and advanced text analysis.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.classifier_simple import SimpleBiasClassifier
from src.core.classifier_enhanced import EnhancedBiasClassifier
from src.utils.advanced_text_analysis import analyze_text_for_bias

def test_basic_vs_enhanced():
    """Compare basic and enhanced analysis on the same text."""
    
    print("=== Enhanced Bias Analysis Demonstration ===\n")
    
    # Test case 1: Post Millennial article (your example)
    test_text_1 = """Former Biden advisor Anita Dunn claims he was 'appropriately accessible to the press' during presidency. 
    Biden was notoriously not available to the press during his term in office. He held the fewest press conferences of any president since the 1980s."""
    
    source_url_1 = "https://thepostmillennial.com/article"
    
    print("📰 Test Case 1: Post Millennial Article")
    print("=" * 50)
    print(f"Text: {test_text_1}")
    print(f"Source: {source_url_1}")
    print()
    
    # Basic analysis
    basic_classifier = SimpleBiasClassifier()
    basic_result = basic_classifier.predict_bias(test_text_1, source_url_1)
    
    # Enhanced analysis
    enhanced_classifier = EnhancedBiasClassifier()
    enhanced_result = enhanced_classifier.predict_bias_enhanced(test_text_1, source_url_1)
    
    # Advanced text analysis only
    advanced_analysis = analyze_text_for_bias(test_text_1)
    
    print("🔍 BASIC ML ANALYSIS:")
    print(f"   Bias: {basic_result.get('bias_label', 'Unknown')}")
    print(f"   Confidence: {basic_result.get('confidence', 0):.1f}%")
    print()
    
    print("🚀 ENHANCED ANALYSIS:")
    print(f"   Basic Bias: {enhanced_result.get('bias_label', 'Unknown')}")
    print(f"   Enhanced Bias: {enhanced_result.get('enhanced_bias_label', 'Unknown')}")
    print(f"   Enhanced Confidence: {enhanced_result.get('enhanced_confidence', 0):.1f}%")
    print(f"   Nuance Score: {enhanced_result.get('nuance_score', 0):.2f}")
    print()
    
    if 'bias_indicators' in enhanced_result:
        indicators = enhanced_result['bias_indicators']
        print("📊 BIAS INDICATORS:")
        print(f"   Emotional Intensity: {indicators.get('emotional_intensity', 0):.2f}")
        print(f"   Loaded Language: {indicators.get('loaded_language_score', 0):.2f}")
        print(f"   Certainty Bias: {indicators.get('certainty_bias', 0):.2f}")
        print(f"   Subjectivity: {indicators.get('overall_subjectivity', 0):.2f}")
        print()
    
    print("🧠 ADVANCED TEXT ANALYSIS:")
    sentiment = advanced_analysis.get('sentiment', {})
    print(f"   Sentiment Polarity: {sentiment.get('polarity', 0):.2f} (-1 to 1)")
    print(f"   Subjectivity: {sentiment.get('subjectivity', 0):.2f} (0 to 1)")
    print(f"   Overall Tone: {sentiment.get('overall_tone', 'Unknown')}")
    print()
    
    loaded_language = advanced_analysis.get('loaded_language', {})
    print("📝 LOADED LANGUAGE DETECTED:")
    for category, data in loaded_language.items():
        if data.get('count', 0) > 0:
            words = data.get('words_found', [])
            print(f"   {category.title()}: {', '.join(words)}")
    print()
    
    factual_claims = advanced_analysis.get('factual_claims', [])
    if factual_claims:
        print("🔍 FACTUAL CLAIMS DETECTED:")
        for i, claim in enumerate(factual_claims, 1):
            print(f"   {i}. \"{claim.get('text', '')[:100]}...\"")
            print(f"      Confidence: {claim.get('confidence', 0):.2f}")
            print(f"      Verifiability: {claim.get('verifiability', 'Unknown')}")
        print()
    
    print("=" * 50)
    print()
    
    # Test case 2: Tariff example (your second example)
    test_text_2 = """Tariffs are not paid by the American people or consumers. This is a common misconception. 
    According to economic experts and studies, tariffs are actually paid by foreign exporters, not American consumers."""
    
    source_url_2 = "https://example.com/article"
    
    print("📰 Test Case 2: Tariff Article")
    print("=" * 50)
    print(f"Text: {test_text_2}")
    print(f"Source: {source_url_2}")
    print()
    
    # Enhanced analysis for tariff example
    enhanced_result_2 = enhanced_classifier.predict_bias_enhanced(test_text_2, source_url_2)
    advanced_analysis_2 = analyze_text_for_bias(test_text_2)
    
    print("🚀 ENHANCED ANALYSIS:")
    print(f"   Enhanced Bias: {enhanced_result_2.get('enhanced_bias_label', 'Unknown')}")
    print(f"   Enhanced Confidence: {enhanced_result_2.get('enhanced_confidence', 0):.1f}%")
    print(f"   Nuance Score: {enhanced_result_2.get('nuance_score', 0):.2f}")
    print()
    
    if 'bias_indicators' in enhanced_result_2:
        indicators_2 = enhanced_result_2['bias_indicators']
        print("📊 BIAS INDICATORS:")
        print(f"   Emotional Intensity: {indicators_2.get('emotional_intensity', 0):.2f}")
        print(f"   Loaded Language: {indicators_2.get('loaded_language_score', 0):.2f}")
        print(f"   Certainty Bias: {indicators_2.get('certainty_bias', 0):.2f}")
        print(f"   Subjectivity: {indicators_2.get('overall_subjectivity', 0):.2f}")
        print()
    
    factual_claims_2 = advanced_analysis_2.get('factual_claims', [])
    if factual_claims_2:
        print("🔍 FACTUAL CLAIMS DETECTED:")
        for i, claim in enumerate(factual_claims_2, 1):
            print(f"   {i}. \"{claim.get('text', '')[:100]}...\"")
            print(f"      Confidence: {claim.get('confidence', 0):.2f}")
            print(f"      Verifiability: {claim.get('verifiability', 'Unknown')}")
        print()
    
    print("=" * 50)
    print()
    
    # Summary
    print("🎯 KEY INSIGHTS:")
    print("1. The enhanced classifier can detect nuanced language patterns")
    print("2. It identifies loaded language, emotional intensity, and certainty bias")
    print("3. It extracts and analyzes factual claims for verifiability")
    print("4. It provides more detailed confidence adjustments based on text analysis")
    print("5. It can distinguish between moderate and extreme bias more accurately")
    print()
    print("💡 This addresses your concern about nuance - the enhanced system can:")
    print("   • Detect subtle framing differences ('not readily available' vs 'appropriately accessible')")
    print("   • Identify factual claims and assess their verifiability")
    print("   • Measure emotional intensity and loaded language")
    print("   • Provide more nuanced bias classifications")

if __name__ == "__main__":
    test_basic_vs_enhanced() 