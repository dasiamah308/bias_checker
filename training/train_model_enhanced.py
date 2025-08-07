import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import sys
import os

# Add parent directory to path to import bias_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bias_utils import get_source_bias_features, extract_domain

def create_sample_dataset():
    """Create a sample dataset for demonstration if labeled_articles.csv doesn't exist."""
    sample_data = {
        'text': [
            "Government announces new climate policy to reduce emissions by 50% by 2030.",
            "Senator criticizes wasteful spending in latest budget proposal.",
            "New study shows economic benefits of renewable energy investment.",
            "Conservative think tank releases report on fiscal responsibility.",
            "Progressive group advocates for universal healthcare coverage.",
            "Bipartisan committee reaches agreement on infrastructure bill.",
            "Liberal media outlet reports on social justice initiatives.",
            "Right-wing commentator discusses traditional family values.",
            "Centrist organization promotes moderate policy solutions.",
            "Environmental activists protest against fossil fuel subsidies."
        ],
        'label': ['Left', 'Right', 'Center', 'Right', 'Left', 'Center', 'Left', 'Right', 'Center', 'Left'],
        'source_url': [
            'https://www.nytimes.com/article1',
            'https://www.foxnews.com/article2', 
            'https://www.reuters.com/article3',
            'https://www.breitbart.com/article4',
            'https://www.huffpost.com/article5',
            'https://www.bbc.com/article6',
            'https://www.vox.com/article7',
            'https://www.nationalreview.com/article8',
            'https://www.thehill.com/article9',
            'https://www.theguardian.com/article10'
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

def train_enhanced_model(df):
    """Train a model using both text and bias features."""
    
    # Extract bias features
    print("Extracting bias features...")
    bias_df = extract_bias_features(df)
    
    # Prepare text features
    print("Preparing text features...")
    text_vectorizer = TfidfVectorizer(
        max_features=5000, 
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Prepare bias features
    bias_numeric_features = [
        'allsides_score_normalized',
        'adfontes_bias_score', 
        'adfontes_reliability_score',
        'score_difference',
        'reliability_score'
    ]
    
    bias_categorical_features = [
        'allsides_label',
        'adfontes_label',
        'agreement_level',
        'consensus_bias'
    ]
    
    # Create feature transformers
    text_transformer = Pipeline([
        ('vectorizer', text_vectorizer)
    ])
    
    bias_numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    bias_categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', LabelEncoder())
    ])
    
    # Combine all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'text'),
            ('bias_numeric', bias_numeric_transformer, bias_numeric_features),
            ('bias_categorical', bias_categorical_transformer, bias_categorical_features)
        ],
        remainder='drop'
    )
    
    # Create full pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Prepare data
    X = pd.concat([df[['text']], bias_df], axis=1)
    y = df['label']
    
    # Split data - handle small datasets
    if len(X) < 15:
        # For very small datasets, use all data for training
        X_train, X_test, y_train, y_test = X, X, y, y
        print("Warning: Small dataset detected. Using all data for training.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Train model
    print("Training enhanced model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Cross-validation - handle small datasets
    if len(X) >= 10:
        cv_folds = min(5, len(X) // 2)  # Ensure we have enough samples per fold
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    else:
        print("Skipping cross-validation due to small dataset size")
    
    return model, text_vectorizer, bias_df

def save_model_and_features(model, text_vectorizer, bias_df, output_dir="../models"):
    """Save the trained model and related components."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full pipeline
    joblib.dump(model, os.path.join(output_dir, "enhanced_model.pkl"))
    
    # Save text vectorizer separately for easy access
    joblib.dump(text_vectorizer, os.path.join(output_dir, "text_vectorizer.pkl"))
    
    # Save bias feature columns for reference
    bias_columns = {
        'numeric_features': [
            'allsides_score_normalized',
            'adfontes_bias_score', 
            'adfontes_reliability_score',
            'score_difference',
            'reliability_score'
        ],
        'categorical_features': [
            'allsides_label',
            'adfontes_label',
            'agreement_level',
            'consensus_bias'
        ]
    }
    joblib.dump(bias_columns, os.path.join(output_dir, "bias_columns.pkl"))
    
    print(f"Model and components saved to {output_dir}/")

def main():
    """Main training function."""
    print("=== Enhanced Bias Detection Model Training ===")
    
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
    model, text_vectorizer, bias_df = train_enhanced_model(df)
    
    # Save model and components
    save_model_and_features(model, text_vectorizer, bias_df)
    
    print("\n=== Training Complete ===")
    print("Enhanced model with bias features has been trained and saved!")
    print("You can now use this model with both text content and source bias information.")

if __name__ == "__main__":
    main() 