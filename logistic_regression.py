# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

class LogisticRegression_model:
    def __init__(self):
        self.model_name = 'LogisticRegression'

    def train_model(self, X_train, X_test, y_train, y_test):
        log_reg = LogisticRegression(solver='lbfgs')
        parameters = {'penalty': ['l2'],
                      'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(estimator=log_reg,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        log_reg = LogisticRegression(solver='lbfgs', **best_params)
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        cross_validation_score = cross_val_score(estimator=log_reg, X=X_train, y=y_train, n_jobs=-1, cv=10)
        acc_score = accuracy_score(y_pred, y_test)
        conf_score = confusion_matrix(y_pred, y_test)
        return acc_score, conf_score, cross_validation_score.mean()


