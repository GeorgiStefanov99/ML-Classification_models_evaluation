from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

class XGBoost_model:
    def __init__(self):
        self.model_name = 'XGBoost'

    def train_model(self, X_train, X_test, y_train, y_test):
        classifier = XGBClassifier()
        parameters = {'learning_rate': [0.01, 0.1, 0.2],
                      'max_depth': [3, 5, 7],
                      'n_estimators': [100, 200, 300]}
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        grid_search.fit(X_train, y_train_encoded)
        best_params = grid_search.best_params_
        xgb_model = XGBClassifier(**best_params)
        xgb_model.fit(X_train, y_train_encoded)
        y_pred = xgb_model.predict(X_test)
        cross_validation_score = cross_val_score(estimator=xgb_model, X=X_train, y=y_train, cv=10, n_jobs=-1)
        acc_score = accuracy_score(y_pred, y_test_encoded)
        conf_score = confusion_matrix(y_pred, y_test_encoded)
        return acc_score, conf_score, cross_validation_score.mean()