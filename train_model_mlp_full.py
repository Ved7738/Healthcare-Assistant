# train_model_mlp_full.py

import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/dataset.csv")
data.fillna("none", inplace=True)

# Filter diseases with at least 3 records
disease_counts = data["Disease"].value_counts()
data = data[data["Disease"].isin(disease_counts[disease_counts > 2].index)]

# All symptom columns
symptom_cols = [col for col in data.columns if col != "Disease"]
all_symptoms = sorted(set(data[symptom_cols].values.ravel()) - {"none"})

# Encode each row to 262-symptom binary vector
def encode(row):
    present = set(row.values) - {"none"}
    return [1 if symptom in present else 0 for symptom in all_symptoms]

X = data[symptom_cols].apply(encode, axis=1, result_type="expand")
X.columns = all_symptoms

# Encode disease labels
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(data["Disease"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("model_mlp.pkl", "wb") as f:
    pickle.dump(model, f)

with open("symptom_encoder.pkl", "wb") as f:
    pickle.dump(all_symptoms, f)

with open("disease_encoder.pkl", "wb") as f:
    pickle.dump(disease_encoder, f)

print("✅ MLP model trained with all 262 symptoms and saved.")
