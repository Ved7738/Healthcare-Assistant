# train_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/dataset.csv")
data.fillna("none", inplace=True)

# Drop rare diseases with very few records (optional)
disease_counts = data["Disease"].value_counts()
data = data[data["Disease"].isin(disease_counts[disease_counts > 2].index)]

# Extract all symptom columns
symptom_cols = [col for col in data.columns if col != "Disease"]

# Get full unique symptoms from all rows
all_symptoms = sorted(set(data[symptom_cols].values.ravel()) - {"none"})

# Convert each row into binary vector of symptoms
def encode_symptoms(row):
    present = set(row.values) - {"none"}
    return [1 if symptom in present else 0 for symptom in all_symptoms]

X = data[symptom_cols].apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms

# Encode target labels
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(data["Disease"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train improved model (Random Forest)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("symptom_encoder.pkl", "wb") as f:
    pickle.dump(all_symptoms, f)

with open("disease_encoder.pkl", "wb") as f:
    pickle.dump(disease_encoder, f)

print("✅ Model trained on ALL symptoms and saved successfully.")
