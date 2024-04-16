# Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV


class SVM:
    def __init__(self):
        self.model_name = 'SVM'

    def train_model(self, X_train, X_test, y_train, y_test):
        svc = SVC()
        parameters = {'C': [0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto']}
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        classifier = SVC(**grid_search.best_params_)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cross_validation_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, n_jobs=-1, cv=10)
        acc_score = accuracy_score(y_pred, y_test)
        conf_score = confusion_matrix(y_pred, y_test)
        return acc_score, conf_score, cross_validation_score.mean()



