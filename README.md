# Breast-Cancer-Detection-SVM
This project utilizes the Support Vector Machine (SVM) algorithm for breast cancer detection. It analyzes medical data to classify tumors as benign or malignant based on key features. By leveraging SVM's ability to handle high-dimensional data, the model enhances diagnostic accuracy, aiding early detection and improving patient outcomes.(The SVM algorithm creates a hyperplane to segregate n-dimensional space into classes and identify the correct category of new data points. The extreme cases that help create the hyperplane are called support vectors, hence the name Support Vector Machine.)This project uses Support Vector Machine (SVM) to classify breast tumors as benign or malignant using the Breast Cancer Wisconsin (Diagnostic) dataset. SVM is effective for binary classification, especially in high-dimensional feature spaces.

📂 Dataset

Features: 30 numeric columns describing tumor characteristics (mean, standard error, worst)

Target: diagnosis → M (Malignant), B (Benign)

Samples: 569

Columns include:
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, ... , fractal_dimension_worst

⚙️ Python Implementation
# Breast Cancer Detection using SVM

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("breast_cancer_data.csv")  # Replace with your CSV file path

# Encode target column (M=1, B=0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM with GridSearch for best parameters
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid.best_params_)

# Predictions
y_pred = grid.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

✅ Key Points

Label Encoding: Converts M and B to numeric labels (1 & 0).

Feature Scaling: Standardization is critical for SVM performance.

Hyperparameter Tuning: C (regularization) and gamma (kernel coefficient) are tuned using GridSearchCV.

Evaluation: Accuracy, precision, recall, F1-score, and confusion matrix.

📊 Expected Results

Accuracy: ~96–98%

High precision and recall for detecting malignant tumors, which is important for minimizing false negatives.
