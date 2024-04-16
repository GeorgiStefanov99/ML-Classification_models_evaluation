# Kernel SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

class KernelSVM:
    def __init__(self):
        self.model_name = 'KernelSVM'

    def train_model(self, X_train, X_test, y_train, y_test):
        svc = SVC()
        parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
                      {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'],
                       'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        cross_validation_score = cross_val_score(estimator=svc, X=X_train, y=y_train, n_jobs=-1, cv=10)
        acc_score = accuracy_score(y_pred, y_test)
        conf_score = confusion_matrix(y_pred, y_test)
        return acc_score, conf_score,cross_validation_score.mean()
#     svc = SVC()
#     parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
#                   {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'],
#                    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#     grid_search = GridSearchCV(estimator=svc,
#                                param_grid=parameters,
#                                scoring='accuracy',
#                                cv=10,
#                                n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     y_pred = grid_search.predict(X_test)
#     cross_validation_scores = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=10)
#     return accuracy_score(y_test, y_pred), cross_validation_scores

