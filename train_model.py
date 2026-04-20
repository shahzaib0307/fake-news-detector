import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from preprocess import clean_text

print("=" * 50)
print("FAKE NEWS DETECTOR — MODEL TRAINING")
print("=" * 50)

# ─────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────
print("\n[1/6] Loading datasets...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Assign labels: 0 = FAKE, 1 = REAL
fake_df['label'] = 0
true_df['label'] = 1

print(f"  Fake articles loaded : {len(fake_df)}")
print(f"  Real articles loaded : {len(true_df)}")

# ─────────────────────────────────────────────────
# 2. COMBINE & SHUFFLE
# ─────────────────────────────────────────────────
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Combined total       : {len(df)}")
print(f"  Label distribution   :\n{df['label'].value_counts()}")

# ─────────────────────────────────────────────────
# 3. FEATURE ENGINEERING — combine title + text
# ─────────────────────────────────────────────────
print("\n[2/6] Combining title and text columns...")
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Check for any remaining nulls
print(f"  Null values in content: {df['content'].isnull().sum()}")

# ─────────────────────────────────────────────────
# 4. TEXT PREPROCESSING
# ─────────────────────────────────────────────────
print("\n[3/6] Cleaning and preprocessing text...")
print("  This may take 2-4 minutes...")
start = time.time()
df['cleaned'] = df['content'].apply(clean_text)
elapsed = time.time() - start
print(f"  Done in {elapsed:.1f} seconds.")

# Show a sample cleaned article
print("\n  Sample original (first 150 chars):")
print(" ", df['content'].iloc[0][:150])
print("  Sample cleaned:")
print(" ", df['cleaned'].iloc[0][:150])

# ─────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────
print("\n[4/6] Splitting into train and test sets (80/20)...")
X = df['cleaned']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y      # ensures equal proportion of fake/real in both splits
)

print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")

# ─────────────────────────────────────────────────
# 6. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────────
print("\n[5/6] Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=50000,   # keep top 50,000 most informative terms
    ngram_range=(1, 2),   # use single words AND word pairs (bigrams)
    sublinear_tf=True,    # apply log normalization to term frequencies
    min_df=2              # ignore terms that appear in fewer than 2 documents
)

X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocabulary + transform
X_test_tfidf  = vectorizer.transform(X_test)         # transform only (no fitting!)

print(f"  Vocabulary size  : {len(vectorizer.vocabulary_)}")
print(f"  Feature matrix   : {X_train_tfidf.shape}")

# ─────────────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────────────
print("\n[6/6] Training Logistic Regression model...")
model = LogisticRegression(
    C=1.0,              # regularization strength (higher = less regularization)
    max_iter=1000,      # max iterations for solver convergence
    solver='lbfgs',     # efficient solver for large datasets
    n_jobs=-1,          # use all CPU cores
    random_state=42
)

start = time.time()
model.fit(X_train_tfidf, y_train)
elapsed = time.time() - start
print(f"  Training complete in {elapsed:.1f} seconds.")

# ─────────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

y_pred      = model.predict(X_test_tfidf)
y_pred_prob = model.predict_proba(X_test_tfidf)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
conf_mat  = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

print("Confusion Matrix:")
print("              Predicted FAKE  Predicted REAL")
print(f"Actual FAKE       {conf_mat[0][0]:>6}          {conf_mat[0][1]:>6}")
print(f"Actual REAL       {conf_mat[1][0]:>6}          {conf_mat[1][1]:>6}")

# Show top words associated with FAKE and REAL
feature_names = vectorizer.get_feature_names_out()
coef          = model.coef_[0]
top_fake_idx  = np.argsort(coef)[:15]
top_real_idx  = np.argsort(coef)[-15:][::-1]

print("\nTop 15 words strongly associated with FAKE news:")
print([feature_names[i] for i in top_fake_idx])

print("\nTop 15 words strongly associated with REAL news:")
print([feature_names[i] for i in top_real_idx])

# ─────────────────────────────────────────────────
# 9. SAVE MODEL & VECTORIZER
# ─────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(model,      'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("\nModel saved      : models/model.pkl")
print("Vectorizer saved : models/vectorizer.pkl")
print("\nTraining complete. You can now run app.py")