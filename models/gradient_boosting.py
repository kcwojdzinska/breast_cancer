from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier


def gradient_boosting(x_train, x_test, y_train, y_test):
    xgboost = XGBClassifier().fit(x_train, y_train)
    predictions_xgboost = xgboost.predict(x_test)
    print(confusion_matrix(y_test, predictions_xgboost))
    accuracy_xgboost = accuracy_score(y_test, predictions_xgboost)
    print("Accuracy XGBoost:", accuracy_xgboost)
