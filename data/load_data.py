import pandas as pd


def load_data():
    train = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/train_set.csv")
    test = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/test_set.csv")
    column_with_labels = 'diagnosis'
    x_train = train.drop([column_with_labels], axis=1)
    x_test = test.drop([column_with_labels], axis=1)
    y_train = train[column_with_labels]
    y_test = test[column_with_labels]
    return x_train, x_test, y_train, y_test
