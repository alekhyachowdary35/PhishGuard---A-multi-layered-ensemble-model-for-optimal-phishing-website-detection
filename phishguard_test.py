import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy.sparse import hstack, csr_matrix
from phishguard_train import extract_lexical_features

print("[*] Loading dataset...")
df = pd.read_csv("phishing_dataset.csv")

# Ensure correct columns
if 'url' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'url' and 'label' columns.")

# Load model artifacts
print("[*] Loading trained model artifacts...")
artifacts = joblib.load("model_output/phishguard_artifacts.joblib")
base_models = joblib.load("model_output/phishguard_base_models.joblib")

tfidf = artifacts['tfidf']
scaler = artifacts['scaler']
meta_clf = artifacts['meta_clf']

# Split dataset for evaluation
from sklearn.model_selection import train_test_split
X_urls_train, X_urls_test, y_train, y_test = train_test_split(
    df['url'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"[*] Testing samples: {len(X_urls_test)}")

# Feature extraction for test URLs
print("[*] Extracting lexical and TF-IDF features...")
lex_test = extract_lexical_features(X_urls_test)
tfidf_test = tfidf.transform(X_urls_test)

X_test_sparse = hstack([tfidf_test, csr_matrix(lex_test.values)], format='csr')

# Get base model predictions (for meta input)
print("[*] Getting base model predictions...")
base_preds = []
for name, model in base_models.items():
    print(f"   - {name}")
    base_pred = model.predict_proba(X_test_sparse)[:, 1].reshape(-1, 1)
    base_preds.append(base_pred)

import numpy as np
X_meta_test = np.hstack(base_preds)

# Scale meta features and predict final output
print("[*] Making final predictions...")
X_meta_scaled = scaler.transform(X_meta_test)
y_pred = meta_clf.predict(X_meta_scaled)

# Evaluate performance
print("\n===== Evaluation Results =====")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
