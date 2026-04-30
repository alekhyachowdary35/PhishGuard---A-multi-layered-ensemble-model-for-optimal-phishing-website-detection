import pandas as pd

# Load the full dataset
df = pd.read_csv("phishing_dataset.csv")

# Take a random sample of 10,000 rows
df = df.sample(n=10000, random_state=42)

# Save as a smaller CSV for quick training
df.to_csv("phishing_sample.csv", index=False)
print("✅ Sample dataset created: phishing_sample.csv")
