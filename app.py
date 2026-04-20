from flask import Flask, render_template, request, jsonify
import joblib
import sqlite3
import os
from datetime import datetime
from preprocess import clean_text

# ─────────────────────────────────────────────────
# INITIALIZE FLASK APP
# ─────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────
# LOAD TRAINED MODEL AND VECTORIZER
# ─────────────────────────────────────────────────
print("Loading model and vectorizer...")

if not os.path.exists('models/model.pkl'):
    raise FileNotFoundError("model.pkl not found. Run train_model.py first.")
if not os.path.exists('models/vectorizer.pkl'):
    raise FileNotFoundError("vectorizer.pkl not found. Run train_model.py first.")

model      = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
print("Model loaded successfully.")

# ─────────────────────────────────────────────────
# DATABASE SETUP — SQLite
# ─────────────────────────────────────────────────
DB_PATH = 'predictions.db'

def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            article    TEXT    NOT NULL,
            prediction TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            fake_prob  REAL    NOT NULL,
            real_prob  REAL    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def save_to_db(article, prediction, confidence, fake_prob, real_prob):
    """Save a prediction result to the database."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        '''INSERT INTO predictions
           (article, prediction, confidence, fake_prob, real_prob, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (
            article[:500],              # store first 500 chars only
            prediction,
            round(confidence, 2),
            round(fake_prob, 2),
            round(real_prob, 2),
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept a JSON body: { "article": "..." }
    Return: { "prediction": "FAKE"/"REAL", "confidence": 97.3,
              "fake_prob": 2.7, "real_prob": 97.3 }
    """
    # 1. Parse request
    data = request.get_json()
    if not data or 'article' not in data:
        return jsonify({'error': 'No article provided. Send JSON with "article" key.'}), 400

    raw_text = data['article'].strip()

    # 2. Validate input length
    if len(raw_text) < 20:
        return jsonify({'error': 'Article is too short. Please provide at least 20 characters.'}), 400
    if len(raw_text) > 100000:
        return jsonify({'error': 'Article too long. Maximum 100,000 characters.'}), 400

    # 3. Preprocess
    cleaned = clean_text(raw_text)
    if len(cleaned.split()) < 3:
        return jsonify({'error': 'After cleaning, not enough meaningful words found.'}), 400

    # 4. Vectorize
    features = vectorizer.transform([cleaned])

    # 5. Predict
    pred_label = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    label      = 'REAL' if pred_label == 1 else 'FAKE'
    fake_prob  = float(pred_proba[0]) * 100
    real_prob  = float(pred_proba[1]) * 100
    confidence = max(fake_prob, real_prob)

    # 6. Save to database
    save_to_db(raw_text, label, confidence, fake_prob, real_prob)

    # 7. Return result
    return jsonify({
        'prediction': label,
        'confidence': round(confidence, 2),
        'fake_prob':  round(fake_prob,  2),
        'real_prob':  round(real_prob,  2)
    })


@app.route('/history')
def history():
    """Return the last 20 predictions from the database."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''
        SELECT article, prediction, confidence, fake_prob, real_prob, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT 20
    ''')
    rows = c.fetchall()
    conn.close()

    results = [
        {
            'article':    r[0],
            'prediction': r[1],
            'confidence': r[2],
            'fake_prob':  r[3],
            'real_prob':  r[4],
            'timestamp':  r[5]
        }
        for r in rows
    ]
    return jsonify(results)


@app.route('/stats')
def stats():
    """Return overall statistics from the database."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('SELECT COUNT(*) FROM predictions')
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'FAKE'")
    total_fake = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'REAL'")
    total_real = c.fetchone()[0]
    c.execute('SELECT AVG(confidence) FROM predictions')
    avg_conf = c.fetchone()[0]
    conn.close()

    return jsonify({
        'total_predictions': total,
        'total_fake':        total_fake,
        'total_real':        total_real,
        'avg_confidence':    round(avg_conf, 2) if avg_conf else 0
    })


# ─────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)