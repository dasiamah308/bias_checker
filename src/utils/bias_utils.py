from ..data.allsides_bias_lookup import allsides_bias
from ..data.adfontes_bias_lookup import adfontes_bias
import re

def extract_domain(url):
    """Extract domain from URL, handling various formats."""
    if not url:
        return None
    
    # Remove protocol
    domain = re.sub(r'^https?://', '', url.lower())
    
    # Remove path and query parameters
    domain = domain.split('/')[0]
    
    # Remove www. prefix
    domain = re.sub(r'^www\.', '', domain)
    
    return domain

def normalize_allsides_to_adfontes(allsides_score):
    """Convert AllSides score (-6 to +6) to Ad Fontes scale (-42 to +42)."""
    # AllSides: -6 to +6 (12 point range)
    # Ad Fontes: -42 to +42 (84 point range)
    # Multiply by 7 to scale up
    return allsides_score * 7

def normalize_adfontes_to_allsides(adfontes_score):
    """Convert Ad Fontes score (-42 to +42) to AllSides scale (-6 to +6)."""
    # Ad Fontes: -42 to +42 (84 point range)
    # AllSides: -6 to +6 (12 point range)
    # Divide by 7 to scale down
    return adfontes_score / 7

def get_bias_info(domain_or_url):
    """Get bias information from both sources if available."""
    domain = extract_domain(domain_or_url)
    if not domain:
        return None
    
    result = {
        'domain': domain,
        'allsides': None,
        'adfontes': None,
        'combined': {}
    }
    
    # Get AllSides data
    if domain in allsides_bias:
        result['allsides'] = allsides_bias[domain]
    
    # Get Ad Fontes data
    if domain in adfontes_bias:
        result['adfontes'] = adfontes_bias[domain]
    
    # Create combined analysis
    if result['allsides'] and result['adfontes']:
        # Normalize scores for comparison
        allsides_normalized = normalize_allsides_to_adfontes(result['allsides']['score'])
        adfontes_score = result['adfontes']['bias_score']
        
        # Calculate agreement/disagreement
        score_difference = abs(allsides_normalized - adfontes_score)
        agreement_level = "High" if score_difference < 5 else "Medium" if score_difference < 10 else "Low"
        
        result['combined'] = {
            'allsides_normalized': round(allsides_normalized, 2),
            'adfontes_score': adfontes_score,
            'score_difference': round(score_difference, 2),
            'agreement_level': agreement_level,
            'reliability_score': result['adfontes']['reliability_score'],
            'consensus_bias': _get_consensus_bias(result['allsides']['label'], result['adfontes']['bias_label'])
        }
    
    return result

def _get_consensus_bias(allsides_label, adfontes_label):
    """Determine consensus bias category between the two sources."""
    # Map labels to 5-category system
    far_left_labels = ['Far Left', 'Hyper-Partisan Left', 'Most Extreme Left']
    lean_left_labels = ['Left', 'Lean Left', 'Skews Left', 'Strong Left']
    center_labels = ['Center', 'Middle or Balanced Bias']
    lean_right_labels = ['Right', 'Lean Right', 'Skews Right', 'Strong Right']
    far_right_labels = ['Far Right', 'Hyper-Partisan Right', 'Most Extreme Right']
    
    allsides_category = None
    adfontes_category = None
    
    if allsides_label and allsides_label in far_left_labels:
        allsides_category = 'Far Left'
    elif allsides_label and allsides_label in lean_left_labels:
        allsides_category = 'Lean Left'
    elif allsides_label and allsides_label in center_labels:
        allsides_category = 'Center'
    elif allsides_label and allsides_label in lean_right_labels:
        allsides_category = 'Lean Right'
    elif allsides_label and allsides_label in far_right_labels:
        allsides_category = 'Far Right'
    
    if adfontes_label and adfontes_label in far_left_labels:
        adfontes_category = 'Far Left'
    elif adfontes_label and adfontes_label in lean_left_labels:
        adfontes_category = 'Lean Left'
    elif adfontes_label and adfontes_label in center_labels:
        adfontes_category = 'Center'
    elif adfontes_label and adfontes_label in lean_right_labels:
        adfontes_category = 'Lean Right'
    elif adfontes_label and adfontes_label in far_right_labels:
        adfontes_category = 'Far Right'
    
    # Return consensus
    if allsides_category == adfontes_category:
        return allsides_category
    elif allsides_category and adfontes_category:
        return f"Mixed ({allsides_category}/{adfontes_category})"
    elif allsides_category:
        return allsides_category
    elif adfontes_category:
        return adfontes_category
    else:
        return "Unknown"

def get_source_bias_features(domain_or_url):
    """Get bias features for machine learning model."""
    bias_info = get_bias_info(domain_or_url)
    if not bias_info:
        return None
    
    features = {
        'domain': bias_info['domain'],
        'has_allsides': bias_info['allsides'] is not None,
        'has_adfontes': bias_info['adfontes'] is not None
    }
    
    # Add AllSides features
    if bias_info['allsides']:
        features.update({
            'allsides_label': bias_info['allsides']['label'],
            'allsides_score': bias_info['allsides']['score'],
            'allsides_score_normalized': normalize_allsides_to_adfontes(bias_info['allsides']['score'])
        })
    
    # Add Ad Fontes features
    if bias_info['adfontes']:
        features.update({
            'adfontes_label': bias_info['adfontes']['bias_label'],
            'adfontes_bias_score': bias_info['adfontes']['bias_score'],
            'adfontes_reliability_score': bias_info['adfontes']['reliability_score']
        })
    
    # Add combined features
    if bias_info['combined']:
        features.update({
            'score_difference': bias_info['combined']['score_difference'],
            'agreement_level': bias_info['combined']['agreement_level'],
            'consensus_bias': bias_info['combined']['consensus_bias'],
            'reliability_score': bias_info['combined']['reliability_score']
        })
    
    return features

# Example usage and testing
if __name__ == "__main__":
    # Test with a few domains
    test_domains = ["foxnews.com", "nytimes.com", "reuters.com", "breitbart.com"]
    
    for domain in test_domains:
        print(f"\n=== {domain} ===")
        bias_info = get_bias_info(domain)
        if bias_info:
            print(f"AllSides: {bias_info['allsides']}")
            print(f"Ad Fontes: {bias_info['adfontes']}")
            if bias_info['combined']:
                print(f"Combined: {bias_info['combined']}")
        else:
            print("No bias data found") 