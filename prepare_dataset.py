import pandas as pd

# Load your CSV
df = pd.read_csv("phishing_dataset.csv")

# Rename columns to match the PhishGuard script
df.rename(columns={
    'URL': 'url',         # change 'URL' to 'url'
    'Class': 'label'      # change 'Class' or 'target' to 'label'
}, inplace=True)

# Save fixed CSV
df.to_csv("phishing_dataset.csv", index=False)
print("✅ CSV columns fixed: url, label")
