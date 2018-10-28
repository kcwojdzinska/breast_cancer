from data.load_data import load_data
from models.logistic_regression import model_logistic_regression
from models.gradient_boosting import model_gradient_boosting

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model_logistic_regression(x_train, x_test, y_train, y_test)
    model_gradient_boosting(x_train, x_test, y_train, y_test)
