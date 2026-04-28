# 📰 Fake News Detector

A machine learning–powered web application that classifies news articles as **FAKE** or **REAL** with a confidence score.

---

## 🚀 Features

- Classifies any news article as **FAKE** or **REAL**
- Returns per-class probabilities and a confidence score
- Stores prediction history in a local SQLite database
- REST API for programmatic access
- Simple web UI served via Flask

---

## 🧠 How It Works

1. **Text Preprocessing** — Raw article text is cleaned: lowercased, URLs/HTML stripped, punctuation removed, stopwords filtered out.
2. **TF-IDF Vectorization** — Cleaned text is converted into numerical features (up to 50,000 unigrams + bigrams).
3. **Logistic Regression** — A trained classifier predicts the label and outputs class probabilities.

---

## 🗂️ Project Structure

```
fake-news-detector/
├── data/
│   ├── Fake.csv          # Labeled fake news articles
│   └── True.csv          # Labeled real news articles
├── models/
│   ├── model.pkl         # Trained Logistic Regression model
│   └── vectorizer.pkl    # Fitted TF-IDF vectorizer
├── static/               # Static assets (CSS, JS)
├── templates/
│   └── index.html        # Web UI
├── app.py                # Flask web server
├── train_model.py        # Model training script
├── preprocess.py         # Text cleaning pipeline
├── predictions.db        # SQLite database (auto-created)
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/shahzaib0307/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

> Requires `data/Fake.csv` and `data/True.csv` to be present.

```bash
python train_model.py
```

This will:
- Load and preprocess the datasets
- Train a Logistic Regression classifier
- Print evaluation metrics (accuracy, ROC-AUC, classification report)
- Save `models/model.pkl` and `models/vectorizer.pkl`

### 4. Start the web server

```bash
python app.py
```

The app will be available at **http://localhost:5000**

---

## 🌐 API Reference

### `POST /predict`

Classify a news article.

**Request body (JSON):**

```json
{
  "article": "Your news article text here..."
}
```

**Response:**

```json
{
  "prediction": "FAKE",
  "confidence": 94.72,
  "fake_prob": 94.72,
  "real_prob": 5.28
}
```

**Constraints:**
- Minimum length: 20 characters
- Maximum length: 100,000 characters

---

### `GET /history`

Returns the last 20 predictions.

**Response:**

```json
[
  {
    "article": "Article text (first 500 chars)...",
    "prediction": "REAL",
    "confidence": 88.5,
    "fake_prob": 11.5,
    "real_prob": 88.5,
    "timestamp": "2026-04-28T17:00:00+00:00"
  }
]
```

---

### `GET /stats`

Returns aggregate prediction statistics.

**Response:**

```json
{
  "total_predictions": 42,
  "total_fake": 18,
  "total_real": 24,
  "avg_confidence": 91.3
}
```

---

## 🛠️ Tech Stack

| Layer        | Technology                         |
|--------------|------------------------------------|
| Language     | Python 3                           |
| ML / NLP     | scikit-learn, NLTK, pandas, NumPy  |
| Web Framework| Flask                              |
| Database     | SQLite                             |
| Serialization| joblib                             |

---

## 📊 Model Details

| Property        | Value                          |
|-----------------|--------------------------------|
| Algorithm       | Logistic Regression            |
| Vectorizer      | TF-IDF (unigrams + bigrams)    |
| Max features    | 50,000                         |
| Train/Test split| 80% / 20%                      |
| Solver          | L-BFGS                         |

---

## 📄 License

This project is open source. Feel free to use and modify it.
