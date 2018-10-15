import pandas as pd


def prepare_dataset_to_split(ratio=0.8):
    data = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/processed_data.csv")
    column_with_labels = 'diagnosis'
    sorted_data = data.sort_values(by=[column_with_labels])
    labels = data[column_with_labels]
    M = [label for label in labels if label != 0]
    B = [label for label in labels if label == 0]
    sum_of_samples_in_dataset = len(M) + len(B)
    perc_of_M_samples_in_dataset = round(len(M)/(len(M)+len(B)),2)
    len_train_set = round(ratio * sum_of_samples_in_dataset)
    num_of_M_samples_in_train_set = round(perc_of_M_samples_in_dataset * len_train_set)
    num_of_B_samples_in_train_set = len_train_set - num_of_M_samples_in_train_set
    B_dataframe = sorted_data.iloc[:len(B), :]
    M_dataframe = sorted_data.iloc[len(B):, :]
    B_dataframe_train = B_dataframe.iloc[:num_of_B_samples_in_train_set:, :]
    B_dataframe_test = B_dataframe.iloc[num_of_B_samples_in_train_set:, :]
    M_dataframe_train = M_dataframe.iloc[:num_of_M_samples_in_train_set, :]
    M_dataframe_test = M_dataframe.iloc[num_of_M_samples_in_train_set:, :]
    train_frame = [B_dataframe_train, M_dataframe_train]
    test_frame = [B_dataframe_test, M_dataframe_test]
    train = pd.concat(train_frame)
    train.sample(frac=1)
    test = pd.concat(test_frame)
    test.sample(frac=1)
    train.to_csv('train_set.csv', encoding='utf-8', index=False)
    test.to_csv('test_set.csv', encoding='utf-8', index=False)
    return train, test
