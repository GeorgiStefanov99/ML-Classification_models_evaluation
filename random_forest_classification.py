# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

class RandomForestClassifier_model:
    def __init__(self):
        self.model_name = 'RandomForestClassification'

    def train_model(self, X_train, X_test, y_train, y_test):
        classifier = RandomForestClassifier()
        parameters = {'n_estimators': [100, 200, 300],
                      'max_depth': [10, 20, 30, None],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        classifier = RandomForestClassifier(**grid_search.best_params_)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cross_validation_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, n_jobs=-1, cv=10)
        acc_score = accuracy_score(y_pred, y_test)
        conf_score = confusion_matrix(y_pred, y_test)
        return acc_score, conf_score, cross_validation_score.mean()


