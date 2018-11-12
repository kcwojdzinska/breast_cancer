from sklearn.linear_model import LogisticRegression
from data.load_data import load_data
from models.gradient_boosting import model_gradient_boosting
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from sklearn import preprocessing
from experiments.feature_importance import find_important_features

def model_logistic_regression(sample):
    x_train, x_test, y_train, y_test = load_data()
    logistic_regression = LogisticRegression()
    fit_log_reg = logistic_regression.fit(x_train, y_train)
    prediction = fit_log_reg.predict(sample)
    prob = fit_log_reg.predict_proba(sample)
    return prediction, prob


def plot_sample():
    df = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/processed_data.csv")
    labels = df.diagnosis
    important_features = df.loc[:, ['area_se', 'texture_mean', 'texture_worst', 'concave points_worst', 'concavity_worst',
                                    'smoothness_worst', 'perimeter_worst', 'smoothness_mean', 'concave points_mean', 'area_worst']]
    to_concat = [labels, important_features]
    data_visualization = pd.concat(to_concat, axis=1)
    plt.figure(figsize=(10,10))
    data = pd.melt(data_visualization,id_vars="diagnosis",var_name="features",value_name='value')
    plt.xticks(rotation=90)
    seaborn.swarmplot(x='features', y='value', hue='diagnosis', data=data)
    plt.show()


if __name__=='__main__':
    path_to_sample='/Users/karola/PycharmProjects/breast_cancer/data/test_samples/sample.csv'
    sample = pd.read_csv(path_to_sample)
    new_sample = np.ravel(sample.values[0, :])
    test_sample = new_sample.reshape(1, -1)

    pred, prob = model_logistic_regression(test_sample)
    plot_sample()
