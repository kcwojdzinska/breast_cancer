from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_curve, roc_auc_score, auc
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



def model_logistic_regression(x_train, x_test, y_train, y_test):
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=4000)
    fit_log_reg = logistic_regression.fit(x_train, y_train)
    predictions = fit_log_reg.predict(x_test)
    misclassified = np.where(predictions != y_test)
    print(y_test[84])
    print(misclassified)
    print(predictions[84])
    print(confusion_matrix(y_test,predictions))
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print("Accuracy Logistic Regression:", accuracy)
    print("Precision Logistic Regression:", precision)
    print("Recall Logistic Regression:", recall)
    f1 = (2 * precision * recall)/(precision+recall)
    print("F1:", f1)
    return logistic_regression
