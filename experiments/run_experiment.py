from data.split_dataset import prepare_dataset_to_split
from models.logistic_regression import logistic_regression
from models.gradient_boosting import gradient_boosting

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_dataset_to_split()
    logistic_regression(x_train, x_test, y_train, y_test)
    gradient_boosting(x_train, x_test, y_train, y_test)
