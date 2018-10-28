import pandas as pd
from data.load_data import load_data
from models.gradient_boosting import model_gradient_boosting
from models.logistic_regression import model_logistic_regression

data = pd.read_csv('/Users/karola/PycharmProjects/breast_cancer/data/datasets/processed_data.csv')
features = data.drop(columns=['diagnosis'])
def find_important_features(model):
    feature_imp_DF = pd.DataFrame({'feature': list(features.columns), 'importance': model.feature_importances_})

    print("Feature Importance:\n")
    print(feature_imp_DF.sort_values(by=['importance'], ascending=False))

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model = model_gradient_boosting(x_train, x_test, y_train, y_test)
    find_important_features(model)