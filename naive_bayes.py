# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

class NaiveBayes:
    def __init__(self):
        self.model_name = 'NaiveBayes'

    def train_model(self, X_train, X_test, y_train, y_test):
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        cross_validation_score = cross_val_score(estimator=nb, X=X_train, y=y_train, n_jobs=-1, cv=10)
        acc_score = accuracy_score(y_pred, y_test)
        conf_score = confusion_matrix(y_pred, y_test)
        return acc_score, conf_score, cross_validation_score.mean()
