import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# Load your CSV into a table-like structure
data = pd.read_csv('landmarks.csv')

# Split into inputs (X = the 63 landmark numbers) and output (y = the letter)
X = data.drop('letter', axis=1)
y = data['letter']

# Normalize EVERY sample


# Split into training data and a held-out test set the model never sees during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check how well it actually does on unseen data
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nPer-letter breakdown:\n", classification_report(y_test, predictions))

# Save the trained model to a file so we don't have to retrain every time
joblib.dump(model, 'asl_letter_model.pkl')
print("Saved to asl_letter_model.pkl")