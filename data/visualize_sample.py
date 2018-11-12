import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn


def visualize_one_sample(path_to_sample):
    path_to_data = "/Users/karola/PycharmProjects/breast_cancer/data/datasets/data.csv"
    df, important_features = split_dataframe(path_to_data)
    df_sample, important_features_sample = split_dataframe(path_to_sample)
    labels = df.diagnosis
    concat_data = pd.concat([important_features, important_features_sample])
    standarized_data = (concat_data - concat_data.mean()) / (concat_data.std())
    df1 = standarized_data[:-1]
    df1_sample = standarized_data.iloc[-1]
    df1_sample['diagnosis'] = 'test_sample'
    df1_sample = df1_sample.to_frame()
    df1_sample = df1_sample.T
    to_concat = [df1, labels]
    data_visualization = pd.concat(to_concat, axis=1)
    plt.figure(figsize=(10,10))
    data = pd.melt(data_visualization,id_vars="diagnosis",var_name="features",value_name='value')
    data_sample = pd.melt(df1_sample,id_vars="diagnosis",var_name="features",value_name='value')
    plt.xticks(rotation=90)
    seaborn.swarmplot(x='features', y='value', hue='diagnosis', data=data).set_title('Distribution of most important '
                                                                                     'features and test sample values')
    seaborn.swarmplot(x='features', y='value', hue='diagnosis', data=data_sample, color='red')
    plt.savefig("/Users/karola/PycharmProjects/breast_cancer/src/flask-app/static/output_plot_to_display.png",
                bbox_inches='tight', dpi=500)


def split_dataframe(path_to_dataframe):
    df = pd.read_csv(path_to_dataframe)
    important_features = df[['area_se', 'texture_mean', 'texture_worst', 'concave points_worst', 'concavity_worst',
                             'smoothness_worst', 'perimeter_worst', 'smoothness_mean', 'concave points_mean', 'area_worst']]
    return df, important_features


if __name__ == '__main__':
    visualize_one_sample('/Users/karola/PycharmProjects/breast_cancer/src/flask-app/uploaded/sample.csv')
