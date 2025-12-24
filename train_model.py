import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# 1. Generate Synthetic Data (Simulating Credit Card Transactions)
# Features: amount, time, v1, v2... (Generic features like the famous Kaggle dataset)
print("Generating synthetic data...")
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=10000, 
    n_features=5, 
    n_informative=3, 
    n_redundant=0, 
    n_clusters_per_class=1, 
    weights=[0.99, 0.01], # 99% legitimate, 1% fraud (Class Imbalance)
    random_state=42
)

# Convert to DataFrame for easier handling
feature_names = ["amount", "time_delta", "merchant_score", "location_score", "device_score"]
df = pd.DataFrame(X, columns=feature_names)
df['is_fraud'] = y

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['is_fraud'], test_size=0.2, random_state=42
)

# 3. Handle Imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
# This creates fake "fraud" cases so the model learns better
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 4. Train the Model (Random Forest is robust and easy to demo)
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 5. Evaluate
predictions = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, predictions))

# 6. Save the Model
joblib.dump(model, "fraud_model.joblib")
print("Model saved as 'fraud_model.joblib'")