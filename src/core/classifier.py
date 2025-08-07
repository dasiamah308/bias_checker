import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model
model = joblib.load("models/saved_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_bias(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction  # e.g., "Left", "Center", "Right"
