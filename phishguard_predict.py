import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from phishguard_train import extract_lexical_features

# Load artifacts
artifacts = joblib.load("model_output/phishguard_artifacts.joblib")
base_models = joblib.load("model_output/phishguard_base_models.joblib")

tfidf = artifacts['tfidf']
scaler = artifacts['scaler']
meta_clf = artifacts['meta_clf']

print("🚀 PhishGuard URL Predictor Ready!")

while True:
    url = input("\nEnter a URL (or type 'exit' to quit): ").strip()
    if url.lower() == 'exit':
        break

    # Feature extraction
    df = pd.DataFrame({'url': [url]})
    lex_features = extract_lexical_features(df['url'])
    tfidf_features = tfidf.transform(df['url'])
    X_input = hstack([tfidf_features, csr_matrix(lex_features.values)], format='csr')

    # Get base model predictions
    base_preds = []
    for name, model in base_models.items():
        base_pred = model.predict_proba(X_input)[:, 1].reshape(-1, 1)
        base_preds.append(base_pred)

    import numpy as np
    X_meta = np.hstack(base_preds)
    X_meta_scaled = scaler.transform(X_meta)
    y_pred = meta_clf.predict(X_meta_scaled)[0]

    if y_pred == 1:
        print("✅  Legitimate Website.")
    else:
        print("⚠️  Phishing Website Detected!")
