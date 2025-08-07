#!/usr/bin/env python3
"""
Main entry point for the Bias Checker application.

This script provides a command-line interface for bias detection and analysis.
"""

import argparse
import sys
from src.core import SimpleBiasClassifier, predict_bias, extract_text_from_url

def main():
    parser = argparse.ArgumentParser(
        description="Bias Checker - Analyze political bias in news articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --text "Government announces new climate policy"
  python main.py --url "https://example.com/article"
  python main.py --text "Article text" --source "https://nytimes.com/article"
        """
    )
    
    parser.add_argument(
        '--text', 
        type=str, 
        help='Text content to analyze for bias'
    )
    
    parser.add_argument(
        '--url', 
        type=str, 
        help='URL to scrape and analyze for bias'
    )
    
    parser.add_argument(
        '--source', 
        type=str, 
        help='Source URL for bias lookup (used with --text)'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Show detailed analysis results'
    )
    
    args = parser.parse_args()
    
    if not args.text and not args.url:
        parser.error("Either --text or --url must be provided")
    
    # Initialize classifier
    try:
        classifier = SimpleBiasClassifier()
    except FileNotFoundError:
        print("Error: Model not found. Please run the training script first:")
        print("python training/train_model_simple.py")
        sys.exit(1)
    
    # Analyze content
    if args.url:
        print(f"Analyzing URL: {args.url}")
        result = classifier.predict_bias_from_url(args.url)
    else:
        print("Analyzing text content...")
        result = classifier.predict_bias(args.text, args.source)
    
    # Display results
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print(f"\n=== Bias Analysis Results ===")
    print(f"Predicted Bias: {result['bias_label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if args.verbose:
        print(f"\nDetailed Results:")
        print(f"Text Length: {result['text_length']} characters")
        print(f"Source Domain: {result['source_bias_info']['domain']}")
        print(f"Source Bias: {result['source_bias_info']['consensus_bias']}")
        print(f"Reliability Score: {result['source_bias_info']['reliability_score']}")
        print(f"Has Source Bias Data: {result['has_source_bias']}")
        
        print(f"\nConfidence by Class:")
        for bias, conf in result['confidence_by_class'].items():
            print(f"  {bias}: {conf:.1%}")
    
    if args.url and 'extracted_text' in result:
        print(f"\nExtracted Text Preview:")
        print(f"{result['extracted_text'][:200]}...")

if __name__ == "__main__":
    main() 