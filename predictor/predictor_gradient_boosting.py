from xgboost import XGBClassifier
from data.load_data import load_data
import pandas as pd
import numpy as np
from sklearn import preprocessing

def predictor_gradient_boosting():
    new_sample = pd.read_csv('/Users/karola/PycharmProjects/breast_cancer/data/test_samples/sample.csv')
    # print(np.array(new_sample.transpose))
    x_train, x_test, y_train, y_test = load_data()

    xgboost = XGBClassifier()

    fit_xgboost = xgboost.fit(x_train, y_train)
    # print(x_test)
    # print(x_test.iloc[1].transpose)
    prediction = fit_xgboost.predict(new_sample)
    print(prediction)
    return prediction


if __name__=='__main__':
    predictor_gradient_boosting()
