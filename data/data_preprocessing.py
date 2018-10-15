import pandas as pd

df = pd.read_csv('/Users/karola/PycharmProjects/breast_cancer/data/datasets/data.csv')
data = df.drop(columns=['id', 'Unnamed: 32'])
data.diagnosis.replace(to_replace=dict(M=1, B=0), inplace=True)
data.to_csv('processed_data.csv', encoding='utf-8', index=False)
