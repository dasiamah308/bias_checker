import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("labeled_articles.csv")  # with columns: text, label

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "../models/saved_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
