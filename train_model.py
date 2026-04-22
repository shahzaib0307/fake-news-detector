# train_model.py
# Uses our from-scratch ml_engine.py — zero sklearn algorithms.

import pandas as pd
import numpy as np
import joblib
import os
import time
from ml_engine import (
    TFIDFVectorizer,
    LogisticRegression,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from preprocess import clean_text

print("=" * 55)
print("FAKE NEWS DETECTOR — TRAINING (from-scratch engine)")
print("=" * 55)

# ── 1. Load Data ──
print("\n[1/6] Loading datasets...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Loaded {len(df)} articles  |  Fake: {len(fake_df)}  Real: {len(true_df)}")

# ── 2. Combine title + text ──
print("\n[2/6] Merging title and text columns...")
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# ── 3. Preprocess ──
print("\n[3/6] Cleaning text (2-4 min)...")
start = time.time()
df['cleaned'] = df['content'].apply(clean_text)
print(f"  Done in {time.time() - start:.1f}s")

# ── 4. Train / Test Split (manual, no sklearn) ──
print("\n[4/6] Splitting 80/20 train/test...")
np.random.seed(42)
indices    = np.random.permutation(len(df))
split_at   = int(0.8 * len(df))
train_idx  = indices[:split_at]
test_idx   = indices[split_at:]

X_train = df['cleaned'].iloc[train_idx].tolist()
X_test  = df['cleaned'].iloc[test_idx].tolist()
y_train = df['label'].iloc[train_idx].tolist()
y_test  = df['label'].iloc[test_idx].tolist()

print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 5. TF-IDF (our implementation) ──
print("\n[5/6] Running custom TF-IDF vectorizer...")
start      = time.time()
vectorizer = TFIDFVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
X_train_m  = vectorizer.fit_transform(X_train)
X_test_m   = vectorizer.transform(X_test)
print(f"  Matrix shape: {X_train_m.shape}  |  Done in {time.time()-start:.1f}s")

# ── 6. Logistic Regression (our implementation) ──
print("\n[6/6] Training custom Logistic Regression...")
start = time.time()
model = LogisticRegression(
    learning_rate=0.1,
    n_iterations=300,
    lambda_reg=0.001,
    batch_size=256,
    verbose=True
)
model.fit(X_train_m, y_train)
print(f"  Training done in {time.time()-start:.1f}s")

# ── Evaluation ──
print("\n" + "=" * 55)
print("EVALUATION ON TEST SET")
print("=" * 55)

y_pred = model.predict(X_test_m)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc * 100:.2f}%\n")
classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print("              Pred FAKE   Pred REAL")
print(f"Actual FAKE     {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"Actual REAL     {cm[1][0]:>6}      {cm[1][1]:>6}")

# ── Save ──
os.makedirs('models', exist_ok=True)
joblib.dump(model,      'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("\nSaved → models/model.pkl")
print("Saved → models/vectorizer.pkl")
print("\nDone. Run app.py to start the web server.")