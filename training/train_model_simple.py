import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

# Add parent directory to path to import bias_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.bias_utils import get_source_bias_features, extract_domain

def create_sample_dataset():
    """Create a sample dataset for demonstration with 5 bias categories."""
    sample_data = {
        'text': [
            # Far Left examples
            "Revolutionary socialist movement calls for complete economic system overhaul.",
            "Radical environmental activists demand immediate fossil fuel ban.",
            "Communist party advocates for complete wealth redistribution.",
            
            # Lean Left examples
            "Government announces new climate policy to reduce emissions by 50% by 2030.",
            "Progressive group advocates for universal healthcare coverage.",
            "Liberal media outlet reports on social justice initiatives.",
            "Environmental activists protest against fossil fuel subsidies.",
            "Social justice advocates demand police reform measures.",
            "Labor unions fight for better working conditions.",
            
            # Center examples
            "Bipartisan committee reaches agreement on infrastructure bill.",
            "Centrist organization promotes moderate policy solutions.",
            "New study shows economic benefits of renewable energy investment.",
            "Business leaders support tax incentives for job creation.",
            "Environmental scientists warn about climate change impacts.",
            "Independent analysis shows mixed results on policy effectiveness.",
            
            # Lean Right examples
            "Senator criticizes wasteful spending in latest budget proposal.",
            "Conservative think tank releases report on fiscal responsibility.",
            "Fiscal conservatives oppose new government spending programs.",
            "Right-wing commentator discusses traditional family values.",
            "Free market advocates push for deregulation measures.",
            "Traditional values group supports family-oriented policies.",
            
            # Far Right examples
            "Ultra-conservative group calls for complete government shutdown.",
            "Extreme nationalist movement advocates for strict immigration bans.",
            "Far-right commentator promotes conspiracy theories about government.",
            "Radical libertarian group demands elimination of all taxes.",
            "Extremist organization calls for complete social system overhaul."
        ],
        'label': [
            'Far Left', 'Far Left', 'Far Left',
            'Lean Left', 'Lean Left', 'Lean Left', 'Lean Left', 'Lean Left', 'Lean Left',
            'Center', 'Center', 'Center', 'Center', 'Center', 'Center',
            'Lean Right', 'Lean Right', 'Lean Right', 'Lean Right', 'Lean Right', 'Lean Right',
            'Far Right', 'Far Right', 'Far Right', 'Far Right', 'Far Right'
        ],
        'source_url': [
            # Far Left sources
            'https://www.jacobinmag.com/article1',
            'https://www.thenation.com/article2',
            'https://www.monthlyreview.org/article3',
            
            # Lean Left sources
            'https://www.nytimes.com/article4',
            'https://www.huffpost.com/article5',
            'https://www.vox.com/article6',
            'https://www.theguardian.com/article7',
            'https://www.motherjones.com/article8',
            'https://www.thenation.com/article9',
            
            # Center sources
            'https://www.reuters.com/article10',
            'https://www.bbc.com/article11',
            'https://www.wsj.com/article12',
            'https://www.thehill.com/article13',
            'https://www.csmonitor.com/article14',
            'https://www.apnews.com/article15',
            
            # Lean Right sources
            'https://www.foxnews.com/article16',
            'https://www.nationalreview.com/article17',
            'https://www.washingtonexaminer.com/article18',
            'https://www.nypost.com/article19',
            'https://www.washingtontimes.com/article20',
            'https://www.freebeacon.com/article21',
            
            # Far Right sources
            'https://www.breitbart.com/article22',
            'https://www.infowars.com/article23',
            'https://www.zerohedge.com/article24',
            'https://www.dailywire.com/article25',
            'https://www.oann.com/article26'
        ]
    }
    return pd.DataFrame(sample_data)

def extract_bias_features(df):
    """Extract bias features from source URLs."""
    bias_features = []
    
    for _, row in df.iterrows():
        source_url = row.get('source_url', '')
        if source_url:
            features = get_source_bias_features(source_url)
            if features:
                bias_features.append(features)
            else:
                # Create default features for unknown sources
                bias_features.append({
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
                })
        else:
            # No source URL provided
            bias_features.append({
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
            })
    
    return pd.DataFrame(bias_features)

def train_simple_model(df):
    """Train a simple model using text and bias features."""
    
    print("Extracting bias features...")
    bias_df = extract_bias_features(df)
    
    print("Preparing text features...")
    # Text features
    text_vectorizer = TfidfVectorizer(
        max_features=1000, 
        stop_words='english',
        ngram_range=(1, 2)
    )
    X_text = text_vectorizer.fit_transform(df['text'])
    
    # Bias features (numeric only for simplicity)
    bias_numeric = bias_df[['allsides_score_normalized', 'adfontes_bias_score', 'reliability_score']].fillna(0)
    scaler = StandardScaler()
    X_bias = scaler.fit_transform(bias_numeric)
    
    # Combine features
    X_combined = np.hstack([X_text.toarray(), X_bias])
    
    # Train model
    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_combined, df['label'])
    
    # Evaluate
    train_score = model.score(X_combined, df['label'])
    print(f"Training accuracy: {train_score:.4f}")
    
    return model, text_vectorizer, scaler, bias_df

def save_model_and_components(model, text_vectorizer, scaler, bias_df, output_dir="../models"):
    """Save the trained model and components."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save components
    joblib.dump(model, os.path.join(output_dir, "simple_model.pkl"))
    joblib.dump(text_vectorizer, os.path.join(output_dir, "text_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "bias_scaler.pkl"))
    
    # Save bias feature info
    bias_info = {
        'numeric_features': ['allsides_score_normalized', 'adfontes_bias_score', 'reliability_score'],
        'sample_bias_features': bias_df.head().to_dict('records')
    }
    joblib.dump(bias_info, os.path.join(output_dir, "bias_info.pkl"))
    
    print(f"Model and components saved to {output_dir}/")

def main():
    """Main training function."""
    print("=== Simple Enhanced Bias Detection Model Training ===")
    
    # Load or create dataset
    try:
        df = pd.read_csv("labeled_articles.csv")
        print(f"Loaded dataset with {len(df)} articles")
    except FileNotFoundError:
        print("labeled_articles.csv not found. Creating sample dataset...")
        df = create_sample_dataset()
        print(f"Created sample dataset with {len(df)} articles")
    
    # Check required columns
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Adding sample source_url column...")
        df['source_url'] = [f'https://example{i}.com/article{i}' for i in range(len(df))]
    
    # Train model
    model, text_vectorizer, scaler, bias_df = train_simple_model(df)
    
    # Save model and components
    save_model_and_components(model, text_vectorizer, scaler, bias_df)
    
    print("\n=== Training Complete ===")
    print("Simple enhanced model with bias features has been trained and saved!")
    print("You can now use this model with both text content and source bias information.")

if __name__ == "__main__":
    main() 