# K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV


class K_NearestNeighbors:
    def __init__(self):
        self.model_name = 'K_NearestNeighbors'

    def train_model(self, X_train, X_test, y_train, y_test):
        classifier = KNeighborsClassifier()
        parameters = {'n_neighbors': [3, 5, 7, 10],
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan']}
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        classifier = KNeighborsClassifier(**grid_search.best_params_)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cross_validation_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, n_jobs=-1, cv=10)
        acc_score = accuracy_score(y_pred, y_test)
        conf_score = confusion_matrix(y_pred, y_test)
        return acc_score, conf_score, cross_validation_score.mean()


