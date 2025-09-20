# Breast-Cancer-Detection-SVM

Overview

This project predicts whether a breast tumor is malignant or benign using a Support Vector Machine (SVM) classifier. SVM is chosen for its ability to handle high-dimensional data and achieve strong classification performance.

The dataset consists of features extracted from digitized images of breast mass samples, capturing shape, size, and texture characteristics of cell nuclei.

# Dataset

| Feature                   | Description                                                 |
| ------------------------- | ----------------------------------------------------------- |
| id                        | Unique identifier for each sample (not used for prediction) |
| diagnosis                 | Target variable: M = Malignant, B = Benign                  |
| radius\_mean              | Mean of distances from center to points on the perimeter    |
| texture\_mean             | Standard deviation of gray-scale values                     |
| perimeter\_mean           | Mean perimeter of the tumor                                 |
| area\_mean                | Mean area of the tumor                                      |
| smoothness\_mean          | Mean of local variation in radius lengths                   |
| compactness\_mean         | $\frac{perimeter^2}{area} - 1$                              |
| concavity\_mean           | Mean severity of concave portions of the contour            |
| concave points\_mean      | Mean number of concave points                               |
| symmetry\_mean            | Mean symmetry of the tumor                                  |
| fractal\_dimension\_mean  | Mean “coastline approximation” – measures complexity        |
| radius\_se                | Standard error of radius                                    |
| texture\_se               | Standard error of texture                                   |
| perimeter\_se             | Standard error of perimeter                                 |
| area\_se                  | Standard error of area                                      |
| smoothness\_se            | Standard error of smoothness                                |
| compactness\_se           | Standard error of compactness                               |
| concavity\_se             | Standard error of concavity                                 |
| concave points\_se        | Standard error of concave points                            |
| symmetry\_se              | Standard error of symmetry                                  |
| fractal\_dimension\_se    | Standard error of fractal dimension                         |
| radius\_worst             | Largest (worst) radius                                      |
| texture\_worst            | Largest texture value                                       |
| perimeter\_worst          | Largest perimeter                                           |
| area\_worst               | Largest area                                                |
| smoothness\_worst         | Largest smoothness value                                    |
| compactness\_worst        | Largest compactness value                                   |
| concavity\_worst          | Largest concavity value                                     |
| concave points\_worst     | Largest number of concave points                            |
| symmetry\_worst           | Largest symmetry value                                      |
| fractal\_dimension\_worst | Largest fractal dimension value                             |

# Why SVM?

Handles high-dimensional data efficiently.
Finds the optimal hyperplane that separates malignant and benign tumors.
Effective for binary classification problems.

# Workflow

1. Data Preprocessing

    Drop id column
    Encode diagnosis (M → 1, B → 0)
    Handle missing values (if any)
    Scale features using StandardScaler

2. Exploratory Data Analysis (EDA)

    Visualize distributions of features
    Analyze correlations to reduce redundant features

3. SVM Model Training

    Split dataset into training and testing sets
    Train SVM classifier with kernel selection (linear, RBF, or polynomial)
    Tune hyperparameters using GridSearchCV

4. Model Evaluation

    Evaluate using accuracy, precision, recall, F1-score, and ROC-AUC
    Analyze confusion matrix to assess false positives/negatives

5. Prediction

    Predict tumor malignancy for new samples based on features.

# Results

Accuracy: 96–98%

High precision and recall for detecting malignant tumors, which is important for minimizing false negatives.
