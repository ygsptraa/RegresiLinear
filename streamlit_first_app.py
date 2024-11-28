import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
data_path ="water_potability.csv"
data = pd.read_csv(data_path)

# Preprocess data
data = data.dropna()  # Drop rows with missing values

# Define features and target variable
X = data.drop("Potability", axis=1)
y = data["Potability"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)
print("Model saved as rf_model.pkl")
