from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve,auc
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import scikitplot as skplt


def model_gradient_boosting(x_train, x_test, y_train, y_test):
    xgboost = XGBClassifier()
    fit_xgboost = xgboost.fit(x_train, y_train)
    predictions_xgboost = fit_xgboost.predict(x_test)

    print(confusion_matrix(y_test, predictions_xgboost))
    accuracy_xgboost = accuracy_score(y_test, predictions_xgboost)
    precision = precision_score(y_test, predictions_xgboost)
    recall = recall_score(y_test, predictions_xgboost)
    print("Accuracy XGBoost:", accuracy_xgboost)
    print("Precision XGBoost:", precision)
    print("Recall XGBoost:", recall)
    f1 = (2 * precision * recall)/(precision+recall)
    print("F1:", f1)
    return xgboost
