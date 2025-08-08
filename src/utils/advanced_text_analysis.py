"""
Advanced Text Analysis for Bias Detection
Handles nuance, context, and factual claim detection
"""

import re
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
import spacy
from collections import Counter

class AdvancedTextAnalyzer:
    def __init__(self):
        """Initialize the advanced text analyzer."""
        # Load spaCy model for NLP (you'll need to install: pip install spacy && python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_text_nuance(self, text: str) -> Dict:
        """
        Analyze text for nuanced bias indicators.
        
        Returns:
            Dict with various analysis metrics
        """
        analysis = {
            'sentiment': self._analyze_sentiment(text),
            'factual_claims': self._extract_factual_claims(text),
            'loaded_language': self._detect_loaded_language(text),
            'context_indicators': self._analyze_context(text),
            'certainty_levels': self._analyze_certainty(text),
            'source_attribution': self._analyze_source_attribution(text)
        }
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment and emotional tone."""
        blob = TextBlob(text)
        
        # Get overall sentiment
        sentiment = blob.sentiment
        
        # Look for specific emotional indicators
        emotional_words = {
            'anger': ['outrageous', 'disgusting', 'appalling', 'shocking', 'infuriating'],
            'fear': ['dangerous', 'threatening', 'alarming', 'concerning', 'worrisome'],
            'disgust': ['repulsive', 'vile', 'disgusting', 'appalling', 'revolting'],
            'surprise': ['shocking', 'stunning', 'amazing', 'incredible', 'unbelievable'],
            'trust': ['reliable', 'trustworthy', 'credible', 'dependable', 'honest'],
            'distrust': ['suspicious', 'questionable', 'dubious', 'unreliable', 'untrustworthy']
        }
        
        emotion_scores = {}
        text_lower = text.lower()
        for emotion, words in emotional_words.items():
            count = sum(1 for word in words if word in text_lower)
            emotion_scores[emotion] = count
        
        return {
            'polarity': sentiment.polarity,  # -1 to 1
            'subjectivity': sentiment.subjectivity,  # 0 to 1
            'emotion_scores': emotion_scores,
            'overall_tone': self._classify_tone(sentiment.polarity, sentiment.subjectivity)
        }
    
    def _extract_factual_claims(self, text: str) -> List[Dict]:
        """Extract and analyze factual claims in the text."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        claims = []
        
        # Look for factual claim patterns
        claim_patterns = [
            r'(?:claims?|says?|stated?|reported?|revealed?|announced?)\s+(?:that\s+)?([^.!?]+)',
            r'(?:according\s+to|per|as\s+reported\s+by)\s+([^.!?]+)',
            r'(?:study|research|data|statistics?|figures?)\s+(?:show|indicate|reveal|demonstrate)\s+([^.!?]+)',
            r'(?:experts?|analysts?|officials?)\s+(?:say|claim|believe|argue)\s+([^.!?]+)'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(1).strip()
                claims.append({
                    'text': claim_text,
                    'type': 'factual_claim',
                    'confidence': self._assess_claim_confidence(claim_text),
                    'verifiability': self._assess_verifiability(claim_text)
                })
        
        return claims
    
    def _detect_loaded_language(self, text: str) -> Dict:
        """Detect loaded language and bias indicators."""
        loaded_indicators = {
            'exaggeration': ['always', 'never', 'everyone', 'nobody', 'completely', 'totally', 'absolutely'],
            'minimization': ['just', 'merely', 'only', 'simply', 'barely', 'hardly'],
            'qualifiers': ['allegedly', 'supposedly', 'reportedly', 'apparently', 'ostensibly'],
            'intensifiers': ['very', 'extremely', 'incredibly', 'absolutely', 'completely'],
            'dismissive': ['so-called', 'alleged', 'purported', 'claimed', 'self-proclaimed'],
            'authoritative': ['clearly', 'obviously', 'evidently', 'undoubtedly', 'certainly'],
            'hedging': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems', 'appears']
        }
        
        results = {}
        text_lower = text.lower()
        
        for category, words in loaded_indicators.items():
            found_words = [word for word in words if word in text_lower]
            results[category] = {
                'words_found': found_words,
                'count': len(found_words),
                'density': len(found_words) / len(text.split())
            }
        
        return results
    
    def _analyze_context(self, text: str) -> Dict:
        """Analyze contextual indicators of bias."""
        context_indicators = {
            'comparative_language': self._find_comparisons(text),
            'temporal_context': self._analyze_temporal_context(text),
            'source_context': self._analyze_source_context(text),
            'framing_devices': self._detect_framing_devices(text)
        }
        
        return context_indicators
    
    def _analyze_certainty(self, text: str) -> Dict:
        """Analyze levels of certainty and confidence in statements."""
        certainty_indicators = {
            'high_certainty': ['definitely', 'certainly', 'absolutely', 'undoubtedly', 'clearly'],
            'medium_certainty': ['probably', 'likely', 'seems', 'appears', 'suggests'],
            'low_certainty': ['maybe', 'perhaps', 'possibly', 'might', 'could'],
            'uncertainty': ['unclear', 'unknown', 'uncertain', 'unclear', 'ambiguous']
        }
        
        results = {}
        text_lower = text.lower()
        
        for level, words in certainty_indicators.items():
            count = sum(1 for word in words if word in text_lower)
            results[level] = count
        
        # Calculate overall certainty score
        total_words = len(text.split())
        certainty_score = (results['high_certainty'] - results['low_certainty'] - results['uncertainty']) / max(total_words, 1)
        
        results['overall_certainty_score'] = certainty_score
        results['certainty_level'] = self._classify_certainty(certainty_score)
        
        return results
    
    def _analyze_source_attribution(self, text: str) -> Dict:
        """Analyze how sources are attributed and quoted."""
        attribution_patterns = {
            'direct_quotes': len(re.findall(r'["""].*?["""]', text)),
            'indirect_quotes': len(re.findall(r'(?:said|stated|claimed|reported)\s+(?:that\s+)?', text, re.IGNORECASE)),
            'anonymous_sources': len(re.findall(r'(?:anonymous|unnamed|unidentified)\s+(?:source|official|person)', text, re.IGNORECASE)),
            'named_sources': len(re.findall(r'(?:according\s+to|per)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text))
        }
        
        return attribution_patterns
    
    def _find_comparisons(self, text: str) -> List[str]:
        """Find comparative language in text."""
        comparison_patterns = [
            r'(?:more|less|better|worse|higher|lower)\s+than',
            r'(?:compared\s+to|in\s+comparison\s+to|versus)',
            r'(?:similar\s+to|different\s+from|unlike)',
            r'(?:the\s+most|the\s+least|the\s+best|the\s+worst)'
        ]
        
        comparisons = []
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                comparisons.append(match.group(0))
        
        return comparisons
    
    def _analyze_temporal_context(self, text: str) -> Dict:
        """Analyze temporal context and timing indicators."""
        temporal_indicators = {
            'past_reference': len(re.findall(r'(?:previously|earlier|before|in\s+the\s+past)', text, re.IGNORECASE)),
            'future_reference': len(re.findall(r'(?:will|going\s+to|plan\s+to|intend\s+to)', text, re.IGNORECASE)),
            'present_focus': len(re.findall(r'(?:currently|now|at\s+present|currently)', text, re.IGNORECASE)),
            'historical_context': len(re.findall(r'(?:historically|traditionally|in\s+history)', text, re.IGNORECASE))
        }
        
        return temporal_indicators
    
    def _analyze_source_context(self, text: str) -> Dict:
        """Analyze how sources are contextualized."""
        source_context = {
            'credible_sources': len(re.findall(r'(?:expert|official|authority|specialist|professional)', text, re.IGNORECASE)),
            'questionable_sources': len(re.findall(r'(?:anonymous|unnamed|unidentified|unknown)', text, re.IGNORECASE)),
            'partisan_sources': len(re.findall(r'(?:conservative|liberal|progressive|right-wing|left-wing)', text, re.IGNORECASE))
        }
        
        return source_context
    
    def _detect_framing_devices(self, text: str) -> List[str]:
        """Detect framing devices that can indicate bias."""
        framing_devices = []
        
        # Look for framing patterns
        framing_patterns = [
            r'(?:so-called|alleged|purported|claimed)\s+([A-Za-z]+)',
            r'(?:clearly|obviously|evidently)\s+([A-Za-z]+)',
            r'(?:surprisingly|shockingly|amazingly)\s+([A-Za-z]+)',
            r'(?:finally|at\s+last|eventually)\s+([A-Za-z]+)'
        ]
        
        for pattern in framing_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                framing_devices.append(match.group(0))
        
        return framing_devices
    
    def _assess_claim_confidence(self, claim_text: str) -> float:
        """Assess the confidence level of a factual claim."""
        # Simple heuristic based on presence of specific words
        confidence_indicators = {
            'high': ['study', 'research', 'data', 'statistics', 'official', 'confirmed'],
            'medium': ['report', 'analysis', 'expert', 'official', 'according'],
            'low': ['alleged', 'claimed', 'supposed', 'rumored', 'anonymous']
        }
        
        claim_lower = claim_text.lower()
        score = 0.5  # Default medium confidence
        
        for level, words in confidence_indicators.items():
            for word in words:
                if word in claim_lower:
                    if level == 'high':
                        score += 0.3
                    elif level == 'medium':
                        score += 0.1
                    elif level == 'low':
                        score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_verifiability(self, claim_text: str) -> str:
        """Assess how verifiable a claim is."""
        verifiable_indicators = ['study', 'data', 'statistics', 'official', 'confirmed', 'documented']
        unverifiable_indicators = ['alleged', 'claimed', 'supposed', 'rumored', 'anonymous', 'unnamed']
        
        claim_lower = claim_text.lower()
        
        verifiable_count = sum(1 for word in verifiable_indicators if word in claim_lower)
        unverifiable_count = sum(1 for word in unverifiable_indicators if word in claim_lower)
        
        if verifiable_count > unverifiable_count:
            return 'high'
        elif unverifiable_count > verifiable_count:
            return 'low'
        else:
            return 'medium'
    
    def _classify_tone(self, polarity: float, subjectivity: float) -> str:
        """Classify the overall tone based on sentiment analysis."""
        if polarity > 0.3:
            if subjectivity > 0.5:
                return 'positive_subjective'
            else:
                return 'positive_objective'
        elif polarity < -0.3:
            if subjectivity > 0.5:
                return 'negative_subjective'
            else:
                return 'negative_objective'
        else:
            if subjectivity > 0.5:
                return 'neutral_subjective'
            else:
                return 'neutral_objective'
    
    def _classify_certainty(self, certainty_score: float) -> str:
        """Classify the certainty level."""
        if certainty_score > 0.1:
            return 'high_certainty'
        elif certainty_score < -0.1:
            return 'low_certainty'
        else:
            return 'medium_certainty'

def analyze_text_for_bias(text: str) -> Dict:
    """
    Main function to analyze text for nuanced bias indicators.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with comprehensive bias analysis
    """
    analyzer = AdvancedTextAnalyzer()
    return analyzer.analyze_text_nuance(text) 