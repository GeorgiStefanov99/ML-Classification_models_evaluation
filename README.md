# Classification Models

This repository contains implementations of various classification models in Python.

## Overview

The main purpose of this project is to compare the performance of different classification algorithms on a given dataset and choose the best one among all. The models included in this project are:

- K-Nearest Neighbors (KNN)
- Kernel Support Vector Machine (SVM)
- Decision Tree Classification
- Logistic Regression
- Naive Bayes
- Random Forest Classification
- Support Vector Machine (SVM)
- XGBoost

## Usage

To use these models, follow these steps:

1. Install the required dependencies by running:
pip install -r requirements.txt

2. Place the .csv file in the main folder. Then, go to `main.py` and on line 21, replace `'Social_Network_Ads.csv'` with the name of your dataset file inside the brackets. For example:
   `dataset = pd.read_csv('Social_Network_Ads.csv')`
   ```Social_Network_Ads.csv is left for testing purposes if you want to see how application perform just run `main.py` ```
  2.1 NOTE: Current implementation does NOT take care of any missing data.

3. Run the `main.py` script to train and evaluate the models on the given dataset:

4. The script will output the best performing model along with its accuracy score, confusion matrix, and cross-validation score.

## Dataset

The models are trained and evaluated on the Wine dataset, which contains various attributes of different types of wine.

## Implementation Details

Each model is implemented as a separate Python class, with a `train_model` method that takes training and test data as input and returns the accuracy score, confusion matrix, and cross-validation score.

## Example

```python
from k_nearest_neighbors import K_NearestNeighbors
from kernel_svm import KernelSVM
from decision_tree_classification import DecisionTreeClassification
from logistic_regression import LogisticRegression_model
from naive_bayes import NaiveBayes
from random_forest_classification import RandomForestClassifier_model
from support_vector_machine import SVM
from xgboost_model import XGBoost_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize classification models
knn, k_svm, dtr, lr, nb, rfc, svm, xgb = (K_NearestNeighbors(), KernelSVM(), DecisionTreeClassification(),
                                       LogisticRegression_model(), NaiveBayes(), RandomForestClassifier_model(), SVM(),
                                       XGBoost_model())

classification_models = (knn, k_svm, dtr, lr, nb, rfc, svm, xgb)

# Load and preprocess dataset
def get_data():
# Import dataset
    dataset = pd.read_csv('Wine.csv')
    
    # Separate features (X) and labels (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Return the preprocessed data
    return X_train, X_test, y_train, y_test

# Train each model on the dataset and get best upon cross validation score
  def get_best_model():
     X_train, X_test, y_train, y_test = get_data()
     trained_models = []
     for model in classification_models:
         acc_score, conf_score, cross_val_score = model.train_model(X_train, X_test, y_train, y_test)
         trained_models.append((f'Model: {model.model_name}\nAccuracy score: {acc_score:.2f}\nConfusion score: '
                                f'{conf_score}\nCross_validation score: {cross_val_score:.2f}'))
         trained_models.sort(key=lambda x: x[3], reverse=True)
     return trained_models[0]

print(get_best_model())
