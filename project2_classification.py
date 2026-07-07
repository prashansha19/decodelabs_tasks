"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ====================== 1. LOAD AND EXPLORE DATASET ======================
print("🤖 Loading Iris Dataset...\n")

iris = load_iris()

# Convert to DataFrame for better readability
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nSpecies Distribution:")
print(df['species'].value_counts())
print("\nStatistical Summary:")
print(df.describe())

# ====================== 2. DATA VISUALIZATION ======================
print("\n📊 Generating Visualizations...")

# Pairplot - Shows relationships between features
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Iris Dataset - Pairplot', y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# ====================== 3. PREPARE DATA ======================
X = iris.data    # Features
y = iris.target  # Labels

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (StandardScaler) - Important as per PDF
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ====================== 4. TRAIN MODELS ======================
print("\n🚀 Training Models...")

# KNN Model (Recommended in PDF)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Decision Tree (For Comparison)
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X_train_scaled, y_train)

# ====================== 5. EVALUATION ======================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{'='*20} {model_name} Results {'='*20}")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model(knn, X_test_scaled, y_test, "KNN Classifier")
evaluate_model(dt, X_test_scaled, y_test, "Decision Tree Classifier")

# ====================== 6. PREDICTION FUNCTION (Bonus) ======================
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    """Predict species for new flower measurements"""
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    prediction = knn.predict(sample_scaled)
    probability = knn.predict_proba(sample_scaled).max()
    
    species = iris.target_names[prediction[0]]
    print(f"\n🌸 Prediction: {species} (Confidence: {probability*100:.1f}%)")

# Test prediction
print("\n🔍 Testing Prediction Function:")
predict_flower(5.1, 3.5, 1.4, 0.2)   # Should be Setosa
predict_flower(6.5, 3.0, 5.2, 2.0)   # Should be Virginica

print("\n✅ Project 2 Completed Successfully!")
