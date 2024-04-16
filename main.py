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

knn, k_svm, dtr, lr, nb, rfc, svm, xgb = (K_NearestNeighbors(), KernelSVM(), DecisionTreeClassification(),
                                          LogisticRegression_model(), NaiveBayes(), RandomForestClassifier_model(), SVM(),
                                          XGBoost_model())
classification_models = (knn, k_svm, dtr, lr, nb, rfc, svm, xgb)


def get_data():
    # Import dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


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
