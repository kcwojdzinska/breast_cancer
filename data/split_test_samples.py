import pandas as pd
import os

def split_test_to_samples():
    path = "/Users/karola/PycharmProjects/breast_cancer/data/datasets/test_samples/"
    data = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/test_set.csv")
    columns = list(data.columns)
    for index, row in data.iterrows():
        row.to_csv('sample' + str(index) + '.csv', encoding='utf-8')


if __name__ == '__main__':
    split_test_to_samples()
