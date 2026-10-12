# TrustMetrics AI
E-Commerce Fake Review & Sentiment Detector built with Flask, scikit-learn, TF-IDF, and SQLite.

## Run in VS Code on Windows
1. Open the project folder in VS Code.
2. Activate your virtual environment.
3. Install packages: `python -m pip install -r requirements.txt`
4. Train the models: `python train_model.py`
5. Start the app: `python app.py`
6. Open http://127.0.0.1:5000 in your browser.

## Dataset note
The included CSV files are small demonstration datasets so the whole pipeline runs immediately. They are NOT research-grade datasets and cannot support real-world accuracy claims. For a stronger academic project, replace them with larger, properly labelled public datasets. Fake review CSV columns: `review_text`, `fake_label` (Fake/Genuine). Sentiment CSV columns: `review_text`, `sentiment` (Positive/Neutral/Negative). Confidence values are model estimates, not proof that a review is fake.
