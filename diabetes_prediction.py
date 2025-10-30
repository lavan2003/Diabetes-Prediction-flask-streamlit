import os
import pandas as pd

# Check if the file exists in the current folder
print("Current working directory:", os.getcwd())

if os.path.exists("diabetes_prediction_dataset.csv"):
    print("✅ File found! Reading it now...")
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    print("File loaded successfully!")
    print("\nFirst 5 rows:")
    print(data.head())
else:
    print("❌ File not found! Please check the name or move the CSV into this folder.")



# Step 2: Data Preprocessing
print("\n--- Data Preprocessing Started ---")

# Drop duplicate rows (if any)
data = data.drop_duplicates()

# Check for missing values
print("\nMissing Values per column:")
print(data.isnull().sum())

# Fill missing numerical values (if any) with column mean
data = data.fillna(data.mean(numeric_only=True))

# Convert categorical columns (like gender, smoking history) into numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':  # if column is text
        data[col] = le.fit_transform(data[col].astype(str))

print("\nConverted all categorical columns into numbers!")
print(data.head())

print("\n--- Data Preprocessing Completed ---")



# Step 3: Model Training and Testing
print("\n--- Model Training Started ---")

# Step 3.1: Separate features (X) and target (y)
X = data.drop("diabetes", axis=1)   # all columns except 'diabetes'
y = data["diabetes"]                # target column

# Step 3.2: Split dataset into training and testing parts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3.3: Import and train the Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("✅ Model training completed!")

# Step 3.4: Make predictions on test data
y_pred = model.predict(X_test)

# Step 3.5: Evaluate model performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("F1-Score:", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred), 3))

print("\n--- Model Training & Testing Completed ---")

