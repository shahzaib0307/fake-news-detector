# ml_engine.py
# ─────────────────────────────────────────────────────────────
# From-scratch implementations of:
#   1. TF-IDF Vectorizer
#   2. Logistic Regression with Gradient Descent
# No sklearn used anywhere in this file.
# ─────────────────────────────────────────────────────────────

import numpy as np
import math
from collections import Counter


# ══════════════════════════════════════════════════════════════
# PART 1 — TF-IDF VECTORIZER
# ══════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    """
    Converts a list of cleaned text strings into a TF-IDF
    numerical matrix from scratch.

    TF  (Term Frequency)      = count(word in doc) / total_words_in_doc
    IDF (Inverse Doc Freq)    = log((1 + N) / (1 + df(word))) + 1
    TF-IDF                    = TF * IDF
    Each row is then L2-normalized so article length doesn't dominate.
    """

    def __init__(self, max_features=30000, ngram_range=(1, 2), min_df=2):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.min_df       = min_df          # ignore words in fewer than N docs
        self.vocabulary_  = {}              # word → column index
        self.idf_         = {}              # word → idf score
        self.feature_names_ = []

    # ── Internal: extract n-grams from a token list ──
    def _get_ngrams(self, tokens):
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams

    # ── Internal: compute TF for one document ──
    def _compute_tf(self, ngrams):
        total = len(ngrams)
        if total == 0:
            return {}
        counts = Counter(ngrams)
        # Sublinear TF scaling: 1 + log(count)
        return {
            word: 1 + math.log(count)
            for word, count in counts.items()
        }

    # ── Internal: L2 normalize a dict of scores ──
    def _l2_normalize(self, score_dict):
        norm = math.sqrt(sum(v ** 2 for v in score_dict.values()))
        if norm == 0:
            return score_dict
        return {k: v / norm for k, v in score_dict.items()}

    # ── fit: learn vocabulary and IDF from training documents ──
    def fit(self, documents):
        N = len(documents)
        print(f"  [TF-IDF] Building vocabulary from {N} documents...")

        # Step 1: document frequency count for every ngram
        doc_freq = Counter()
        tokenized_docs = []

        for doc in documents:
            tokens = doc.split()
            ngrams = self._get_ngrams(tokens)
            unique_ngrams = set(ngrams)
            doc_freq.update(unique_ngrams)
            tokenized_docs.append(ngrams)

        # Step 2: filter by min_df and pick top max_features by doc_freq
        filtered = {
            word: df
            for word, df in doc_freq.items()
            if df >= self.min_df
        }

        # Sort by frequency descending, take top max_features
        top_words = sorted(
            filtered.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_features]

        # Step 3: build vocabulary index
        self.vocabulary_ = {
            word: idx for idx, (word, _) in enumerate(top_words)
        }
        self.feature_names_ = [word for word, _ in top_words]

        # Step 4: compute IDF for each vocabulary word
        vocab_set = set(self.vocabulary_.keys())
        self.idf_ = {}
        for word in vocab_set:
            df = doc_freq[word]
            # Smoothed IDF formula (same as sklearn default)
            self.idf_[word] = math.log((1 + N) / (1 + df)) + 1

        print(f"  [TF-IDF] Vocabulary size: {len(self.vocabulary_)}")
        return self

    # ── transform: convert documents into TF-IDF matrix (2D list) ──
    def transform(self, documents):
        n_docs   = len(documents)
        n_feats  = len(self.vocabulary_)
        # Use a list of dicts for efficiency, convert to numpy at end
        rows = []

        for doc in documents:
            tokens = doc.split()
            ngrams = self._get_ngrams(tokens)
            tf     = self._compute_tf(ngrams)

            # Multiply TF × IDF for each vocabulary word
            score_dict = {}
            for word, tf_val in tf.items():
                if word in self.vocabulary_:
                    score_dict[word] = tf_val * self.idf_[word]

            # L2 normalize
            score_dict = self._l2_normalize(score_dict)
            rows.append(score_dict)

        # Build dense numpy matrix
        matrix = np.zeros((n_docs, n_feats), dtype=np.float32)
        for i, score_dict in enumerate(rows):
            for word, val in score_dict.items():
                if word in self.vocabulary_:
                    col = self.vocabulary_[word]
                    matrix[i, col] = val

        return matrix

    # ── fit_transform: fit then transform in one step ──
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# ══════════════════════════════════════════════════════════════
# PART 2 — LOGISTIC REGRESSION (Binary Classifier)
# ══════════════════════════════════════════════════════════════

class LogisticRegression:
    """
    Binary Logistic Regression trained with Mini-Batch
    Gradient Descent and L2 regularization.

    Forward pass:  z = X·w + b
    Activation:    sigmoid(z) = 1 / (1 + e^(-z))
    Loss:          Binary Cross-Entropy
    Update rule:   w = w - lr * (gradient + lambda*w)
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_iterations=300,
        lambda_reg=0.001,     # L2 regularization (prevents overfitting)
        batch_size=256,       # mini-batch gradient descent
        verbose=True
    ):
        self.lr          = learning_rate
        self.n_iter      = n_iterations
        self.lambda_reg  = lambda_reg
        self.batch_size  = batch_size
        self.verbose     = verbose
        self.weights_    = None
        self.bias_       = 0.0
        self.loss_history_ = []

    # ── Sigmoid activation function ──
    def _sigmoid(self, z):
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ── Binary cross-entropy loss ──
    def _compute_loss(self, y_true, y_pred):
        eps = 1e-9   # avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    # ── Train the model ──
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.array(y, dtype=np.float32)

        # Initialize weights to small random values
        np.random.seed(42)
        self.weights_ = np.zeros(n_features, dtype=np.float32)
        self.bias_    = 0.0

        print(f"  [LR] Training on {n_samples} samples, {n_features} features")
        print(f"  [LR] Iterations: {self.n_iter}, LR: {self.lr}, Batch: {self.batch_size}")

        for iteration in range(self.n_iter):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end     = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                m       = X_batch.shape[0]

                # Forward pass
                z      = X_batch.dot(self.weights_) + self.bias_
                y_pred = self._sigmoid(z)

                # Compute gradients
                error   = y_pred - y_batch
                dw      = (X_batch.T.dot(error) / m) + (self.lambda_reg * self.weights_)
                db      = np.mean(error)

                # Update weights
                self.weights_ -= self.lr * dw
                self.bias_    -= self.lr * db

            # Compute and log loss every 50 iterations
            if self.verbose and (iteration + 1) % 50 == 0:
                z_all   = X.dot(self.weights_) + self.bias_
                p_all   = self._sigmoid(z_all)
                loss    = self._compute_loss(y, p_all)
                preds   = (p_all >= 0.5).astype(int)
                acc     = np.mean(preds == y.astype(int))
                self.loss_history_.append(loss)
                print(f"  [LR] Iter {iteration+1:>4} | Loss: {loss:.4f} | Train Acc: {acc*100:.2f}%")

        return self

    # ── Predict probabilities ──
    def predict_proba(self, X):
        z      = X.dot(self.weights_) + self.bias_
        p_real = self._sigmoid(z)
        p_fake = 1 - p_real
        # Return as (n_samples, 2) array: col0=FAKE prob, col1=REAL prob
        return np.column_stack([p_fake, p_real])

    # ── Predict class labels ──
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════
# PART 3 — EVALUATION METRICS (from scratch)
# ══════════════════════════════════════════════════════════════

def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred, label=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == label) & (y_true == label))
    fp = np.sum((y_pred == label) & (y_true != label))
    fn = np.sum((y_pred != label) & (y_true == label))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def classification_report(y_true, y_pred):
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 46)
    for label, name in [(0, 'FAKE'), (1, 'REAL')]:
        p, r, f = precision_recall_f1(y_true, y_pred, label)
        print(f"{name:<12} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    acc = accuracy_score(y_true, y_pred)
    print("-" * 46)
    print(f"{'Accuracy':<12} {acc:>10.4f}")