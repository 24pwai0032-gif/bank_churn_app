"""
Bank Churn Model Training Script
Run this ONCE to generate model.pkl before starting the backend.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("Churn_Modelling.csv")

print("Dataset shape:", df.shape)
print(df.head(2))

# ─────────────────────────────────────────────
# 2. FEATURE SELECTION
# ─────────────────────────────────────────────
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
TARGET = "Exited"

X = df[FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
categorical_features = ["Geography", "Gender"]
numerical_features   = [f for f in FEATURES if f not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", numerical_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )),
])

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
model_pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
y_pred = model_pipeline.predict(X_test)
print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────────
with open("backend/model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("\n✅ Model saved to backend/model.pkl")
