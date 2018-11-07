from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from data.load_data import load_data


def model_logistic_regression(sample):
    x_train, x_test, y_train, y_test = load_data()
    logistic_regression = LogisticRegression()
    fit_log_reg = logistic_regression.fit(x_train, y_train)
    prediction = fit_log_reg.predict(sample)
    prob = fit_log_reg.predict_proba(sample)
    return prediction, prob


if __name__=='__main__':
    path_to_sample='/Users/karola/PycharmProjects/breast_cancer/data/test_samples/sample.csv'
    sample = pd.read_csv(path_to_sample)
    new_sample = np.ravel(sample.values[0, :])
    test_sample = new_sample.reshape(1, -1)

    pred, prob = model_logistic_regression(test_sample)
    print(pred, prob)
