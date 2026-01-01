import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Load dataset
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print(df.head())

# Step 2: Data preprocessing
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("Data split complete.")

# Step 4: Train Isolation Forest model
print("Training Isolation Forest model...")
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(X_train)

# Save trained model
joblib.dump(model, "model.pkl")

# Step 5: Evaluate model
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1)  # Convert 1→normal, -1→anomaly

print("\nModel Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("Model and scaler saved successfully.")
