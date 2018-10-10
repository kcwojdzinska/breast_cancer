from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def logistic_regression(x_train, x_test, y_train, y_test):
    logistic_regression = LogisticRegression().fit(x_train, y_train)
    predictions = logistic_regression.predict(x_test)
    print(confusion_matrix(y_test,predictions))
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy Logistic Regression:", accuracy)
