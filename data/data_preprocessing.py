import pandas as pd

df = pd.read_csv('data.csv')
data = df.drop(columns=['id', 'Unnamed: 32'])
data.to_csv('processed_data.csv', encoding='utf-8', index=False)
