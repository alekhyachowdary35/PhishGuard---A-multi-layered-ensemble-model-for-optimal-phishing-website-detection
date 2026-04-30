"""
PhishGuard-style ensemble training script
- Expected input: CSV file with columns: "url", "label" (1 = phishing, 0 = benign)
- Produces: trained_stacked_model.joblib and transformer.joblib
Usage:
    python phishguard_train.py --data data/phishing_dataset.csv --out model_output
"""

import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Optional: if you don't have xgboost/catboost, install them (see requirements file)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# ----------------------------
# Feature engineering helpers
# ----------------------------
def extract_lexical_features(url_series: pd.Series) -> pd.DataFrame:
    """
    Compute simple lexical features from URL strings.
    Returns a DataFrame with features.
    """
    def has_ip(url):
        # simple IP-in-domain check
        return 1 if re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0

    def count_subdomains(url):
        # approximate subdomain count
        domain = re.sub(r'^https?://', '', url)
        domain = domain.split('/')[0]
        return domain.count('.')

    def has_at_symbol(url):
        return 1 if '@' in url else 0

    def url_length(url):
        return len(url)

    def num_digits(url):
        return sum(ch.isdigit() for ch in url)

    def num_special(url):
        return sum((not ch.isalnum()) for ch in url)

    data = {
        'url_len': url_series.apply(url_length),
        'num_digits': url_series.apply(num_digits),
        'num_special': url_series.apply(num_special),
        'has_ip': url_series.apply(has_ip),
        'subdomain_count': url_series.apply(count_subdomains),
        'has_at': url_series.apply(has_at_symbol),
        'count_https': url_series.str.count('https').fillna(0).astype(int),
        'count_http': url_series.str.count('http').fillna(0).astype(int),
        'count_login': url_series.str.lower().str.count('login').fillna(0).astype(int),
        'count_secure': url_series.str.lower().str.count('secure').fillna(0).astype(int),
    }
    return pd.DataFrame(data)

# ----------------------------
# Stacking helper
# ----------------------------
def get_oof_predictions(clf, X, y, X_test, n_splits=5, random_state=42):
    """
    Generate out-of-fold predictions for a base classifier for stacking.
    Returns:
        oof_train: shape (n_samples,)
        oof_test_mean: mean of test folds predictions shape (n_test_samples,)
    """
    n_train = X.shape[0]
    n_test = X_test.shape[0]
    oof_train = np.zeros(n_train)
    oof_test = np.zeros((n_test, n_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        clf_fold = clone(clf)
        clf_fold.fit(X_tr, y_tr)
        oof_train[valid_idx] = clf_fold.predict_proba(X_val)[:, 1]
        oof_test[:, i] = clf_fold.predict_proba(X_test)[:, 1]

    oof_test_mean = oof_test.mean(axis=1)
    return oof_train.reshape(-1, 1), oof_test_mean.reshape(-1, 1)

# ----------------------------
# Main training routine
# ----------------------------
def train_phishguard(data_path, out_dir, test_size=0.2, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    print("[*] Loading data...")
    df = pd.read_csv(data_path)
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns")

    # Basic cleaning
    df['url'] = df['url'].astype(str).str.strip()
    df = df.dropna(subset=['url', 'label']).reset_index(drop=True)
    X_urls = df['url']
    y = df['label'].astype(int).values

    print(f"[*] Total samples: {len(df)} (phishing={y.sum()}, benign={len(df)-y.sum()})")

    # Split into train/test
    X_urls_train, X_urls_test, y_train, y_test = train_test_split(X_urls, y, test_size=test_size,
                                                                  stratify=y, random_state=random_state)
    # Lexical features
    print("[*] Extracting lexical features...")
    lex_train = extract_lexical_features(X_urls_train)
    lex_test = extract_lexical_features(X_urls_test)

    # TF-IDF on URL string (char-level)
    print("[*] Fitting TF-IDF (char-level)...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=4000)
    tfidf_train = tfidf.fit_transform(X_urls_train)
    tfidf_test = tfidf.transform(X_urls_test)

    # Combine lexical features + tfidf
    print("[*] Combining features...")
    # Convert lexical features to numpy arrays
    lex_train_np = lex_train.values.astype(float)
    lex_test_np = lex_test.values.astype(float)

    # Stack horizontally:
    # For memory efficiency, we will use sparse hstack for TF-IDF and dense lexical features.
    from scipy.sparse import hstack, csr_matrix
    X_train_sparse = hstack([tfidf_train, csr_matrix(lex_train_np)], format='csr')
    X_test_sparse = hstack([tfidf_test, csr_matrix(lex_test_np)], format='csr')

    # Standardize lexical (dense) part if using dense-only classifiers. For tree-based it's optional.
    # We'll leave as is since tree models handle raw features fine; logistic regression meta-learner benefits from scaling.
    # Convert sparse to dense for models that require dense inputs (CatBoost/XGBoost accept dense; XGBoost can work with sparse)
    X_train = X_train_sparse
    X_test = X_test_sparse

    # Base classifiers
    print("[*] Initializing base classifiers...")
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=random_state, n_jobs=4)
    else:
        print("[!] xgboost not installed; XGBoost will be skipped.")
        xgb_clf = None

    if CatBoostClassifier is not None:
        cb_clf = CatBoostClassifier(iterations=300, verbose=0, random_state=random_state)
    else:
        print("[!] catboost not installed; CatBoost will be skipped.")
        cb_clf = None

    base_clfs = []
    base_names = []
    base_clfs.append(('rf', rf))
    base_names.append('rf')
    if xgb_clf is not None:
        base_clfs.append(('xgb', xgb_clf)); base_names.append('xgb')
    if cb_clf is not None:
        base_clfs.append(('catboost', cb_clf)); base_names.append('catboost')

    # Generate out-of-fold predictions for all base classifiers
    print("[*] Creating stacking features (out-of-fold)...")
    oof_train_list = []
    oof_test_list = []
    X_train_arr = X_train if hasattr(X_train, "toarray") == False else X_train.toarray()  # some models may require dense; but we will pass sparse to clone fit, it's ok for RF and xgb.
    X_test_arr = X_test if hasattr(X_test, "toarray") == False else X_test.toarray()

    # For compatibility, convert sparse to CSR; get_oof_predictions will clone classifiers and call predict_proba.
    X_train_csr = X_train
    X_test_csr = X_test

    for name, clf in base_clfs:
        print(f"    - Processing base classifier: {name}")
        # If classifier doesn't support predict_proba, wrap or skip
        if not hasattr(clf, "predict_proba"):
            raise ValueError(f"Classifier {name} does not support predict_proba required for stacking.")

        # Use get_oof_predictions with sparse matrices: convert to array for sklearn methods
        oof_tr, oof_te = get_oof_predictions(clf, X_train_csr, y_train, X_test_csr, n_splits=5, random_state=random_state)
        oof_train_list.append(oof_tr)
        oof_test_list.append(oof_te)

    # Concatenate meta features
    X_meta_train = np.hstack(oof_train_list)
    X_meta_test = np.hstack(oof_test_list)

    print("[*] Meta features shape:", X_meta_train.shape)

    # Meta-learner (Logistic Regression) with scaling
    print("[*] Training meta-learner (Logistic Regression)...")
    scaler = StandardScaler()
    X_meta_train_scaled = scaler.fit_transform(X_meta_train)
    X_meta_test_scaled = scaler.transform(X_meta_test)

    meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    meta_clf.fit(X_meta_train_scaled, y_train)

    # Evaluate on test
    print("[*] Evaluating on hold-out test set...")
    y_pred_proba = meta_clf.predict_proba(X_meta_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=== Evaluation (Hold-out Test) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts: tfidf, scaler, meta_clf, and optionally the base models retrained on full train+test if desired.
    print("[*] Saving model artifacts...")
    artifacts = {
        'tfidf': tfidf,
        'scaler': scaler,
        'meta_clf': meta_clf,
        'base_clfs_info': base_names,
    }
    joblib.dump(artifacts, os.path.join(out_dir, 'phishguard_artifacts.joblib'))
    print(f"Saved phishguard_artifacts.joblib to {out_dir}")

    # Optionally: train base classifiers on full data and save them too for direct prediction (not required for stacking inference if you only use meta-clf with saved base predictions).
    print("[*] Training base classifiers on full data (tfidf+lex combined)...")
    X_full_sparse = hstack([tfidf.transform(X_urls), csr_matrix(extract_lexical_features(X_urls).values)], format='csr')

    trained_bases = {}
    for name, clf in base_clfs:
        print(f"    - Fitting {name} on full data...")
        clf_full = clone(clf)
        # Some classifiers accept sparse; XGBoost may accept csr or dense; scikit-learn RF accepts sparse as well.
        clf_full.fit(X_full_sparse, y)
        trained_bases[name] = clf_full

    joblib.dump(trained_bases, os.path.join(out_dir, 'phishguard_base_models.joblib'))
    print(f"Saved phishguard_base_models.joblib to {out_dir}")

    print("[*] Done.")
    return

# ----------------------------
# Command-line interface
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PhishGuard-style stacking ensemble for phishing detection.")
    parser.add_argument('--data', required=True, help='Path to CSV dataset (columns: url,label)')
    parser.add_argument('--out', required=False, default='model_output', help='Output directory to save models')
    args = parser.parse_args()

    train_phishguard(args.data, args.out)
