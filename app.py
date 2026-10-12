import os, sqlite3
from datetime import datetime
import joblib
from flask import Flask, render_template, request, jsonify

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "trustmetrics.db")
MODEL_DIR = os.path.join(ROOT, "model")
app = Flask(__name__)

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            fake_label TEXT NOT NULL,
            fake_confidence REAL NOT NULL,
            sentiment TEXT NOT NULL,
            sentiment_confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )""")

def load_models():
    fake_path = os.path.join(MODEL_DIR, "fake_review_model.pkl")
    sentiment_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    if not os.path.exists(fake_path) or not os.path.exists(sentiment_path):
        return None, None
    return joblib.load(fake_path), joblib.load(sentiment_path)

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/analyze")
def analyze():
    review = str((request.get_json(silent=True) or {}).get("review", "")).strip()
    if not review:
        return jsonify(error="Please enter a review first."), 400
    if len(review) > 5000:
        return jsonify(error="Please keep the review under 5,000 characters."), 400
    fake_model, sentiment_model = load_models()
    if fake_model is None or sentiment_model is None:
        return jsonify(error="Models are not trained. In the terminal run: python train_model.py"), 503

    fake_probs = fake_model.predict_proba([review])[0]
    fake_label = str(fake_model.classes_[fake_probs.argmax()])
    fake_conf = float(fake_probs.max() * 100)
    sentiment_probs = sentiment_model.predict_proba([review])[0]
    sentiment = str(sentiment_model.classes_[sentiment_probs.argmax()])
    sentiment_conf = float(sentiment_probs.max() * 100)

    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT INTO analyses
        (review_text, fake_label, fake_confidence, sentiment, sentiment_confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (review, fake_label, fake_conf, sentiment, sentiment_conf,
         datetime.now().strftime("%d %b %Y, %I:%M %p")))
    return jsonify(fake_label=fake_label, fake_confidence=round(fake_conf, 1),
                   sentiment=sentiment, sentiment_confidence=round(sentiment_conf, 1),
                   note="Demo model estimates only; this is not proof that a review is fake.")

@app.route("/history")
def history():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 50").fetchall()
    return render_template("history.html", rows=rows)

@app.post("/history/clear")
def clear_history():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM analyses")
    return jsonify(message="History cleared")

if __name__ == "__main__":
    app.run(debug=True)
