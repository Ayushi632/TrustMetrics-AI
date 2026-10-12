import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dataset")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(csv_file, label_column, output_file):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    df = df.dropna(subset=["review_text", label_column])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0]
    if df[label_column].nunique() < 2:
        raise ValueError(f"{csv_file} must contain at least two different labels.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"], df[label_column], test_size=0.25,
        random_state=42, stratify=df[label_column]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"\n{output_file} demo validation accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(pipeline, os.path.join(MODEL_DIR, output_file))
    print(f"Saved model to model/{output_file}")

if __name__ == "__main__":
    train_model("fake_reviews_demo.csv", "fake_label", "fake_review_model.pkl")
    train_model("sentiment_reviews_demo.csv", "sentiment", "sentiment_model.pkl")
    print("\nBoth starter models trained. Demo scores are not real-world accuracy claims.")
