import pandas as pd
from data.load_data import load_data
from models.gradient_boosting import model_gradient_boosting
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/karola/PycharmProjects/breast_cancer/data/datasets/processed_data.csv')
features = data.drop(columns=['diagnosis'])


def find_important_features(model):
    feature_imp_DF = pd.DataFrame({'feature': list(features.columns), 'importance': model.feature_importances_})
    importance = feature_imp_DF.sort_values(by=['importance'], ascending=False)
    return importance


def donut_plot(df):
    labels = df['feature']
    sizes = df['importance']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
    centre_circle = plt.Circle((0, 0), 0.75, color='black', fc='white', linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model = model_gradient_boosting(x_train, x_test, y_train, y_test)
    features_imp = find_important_features(model)
    donut_plot(features_imp)
