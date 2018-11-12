import pandas as pd
import os


def split_test_to_samples():
    data = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/test_set.csv")
    dropped_labels = data.drop(columns='diagnosis')
    for index, row in dropped_labels.iterrows():
        transposed = row.to_frame()
        transposed = transposed.T
        transposed.to_csv('sample' + str(index) + '.csv', encoding='utf-8')


if __name__ == '__main__':
    split_test_to_samples()
