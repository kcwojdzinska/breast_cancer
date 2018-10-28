from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score


def model_logistic_regression(x_train, x_test, y_train, y_test):
    logistic_regression = LogisticRegression()
    fit_log_reg = logistic_regression.fit(x_train, y_train)
    predictions = fit_log_reg.predict(x_test)
    print(confusion_matrix(y_test,predictions))
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print("Accuracy Logistic Regression:", accuracy)
    print("Precision Logistic Regression:", precision)
    print("Recall Logistic Regression:", recall)
    return logistic_regression
