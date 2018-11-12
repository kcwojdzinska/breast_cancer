import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn
import lime
import lime.lime_tabular
from sklearn.linear_model import LogisticRegression
from config import UPLOAD_FOLDER

from flask import Flask, request, render_template, url_for, after_this_request, flash, session
from werkzeug.utils import redirect

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.errorhandler(404)
def page_not_found(e):
    return render_template('not_found.html')


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('internal_server_error.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    remove_files_in_dir(UPLOAD_FOLDER)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template(request.url)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('predict', filename=file.filename))
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    # with open("/Users/karola/PycharmProjects/breast_cancer/src/flask-app/uploaded/sample.csv", "r+") as f:
    #     content = f.read()
    #     prediction, probability = predictor_log_reg(content)
    #     prediction = map_prediction_to_string(prediction)
    #     f.close()
    # remove_files_in_dir(UPLOAD_FOLDER)
    remove_files_in_dir_png('/Users/karola/PycharmProjects/breast_cancer/src/flask-app/static')
    files = os.listdir(UPLOAD_FOLDER)
    possible_path = [os.path.join(UPLOAD_FOLDER, file) for file in files][0]
    sample = preprocess_sample(possible_path)
    prediction, probability = predictor_log_reg(sample)
    prediction = map_prediction_to_string(prediction)
    probability = find_probability(probability)
    visualize_one_sample(possible_path)
    # redirect(url_for('lime'))
    return render_template('prediction.html', prediction=prediction, probability=probability)


@app.route('/lime_output', methods=['GET', 'POST'])
def lime_output():
    return render_template('lime_output.html', title='Further analysis')


def preprocess_sample(path_to_sample):
    sample = pd.read_csv(path_to_sample)
    new_sample = np.ravel(sample.values[0, :])
    test_sample = new_sample.reshape(1, -1)
    return test_sample


def predictor_log_reg(sample):
    x_train, x_test, y_train, y_test = load_data()
    logistic_regression = LogisticRegression()
    fit_log_reg = logistic_regression.fit(x_train, y_train)
    prediction = fit_log_reg.predict(sample)
    probability = fit_log_reg.predict_proba(sample)
    return prediction, probability

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
    plt.figure(figsize=(15,15))
    data = pd.melt(data_visualization,id_vars="diagnosis",var_name="features",value_name='value')
    data_sample = pd.melt(df1_sample,id_vars="diagnosis",var_name="features",value_name='value')
    plt.xticks(rotation=90)
    seaborn.swarmplot(x='features', y='value', hue='diagnosis', data=data).set_title('Distribution of most important '
                                                                                     'features and test sample values')
    seaborn.set_palette('bright')
    seaborn.swarmplot(x='features', y='value', hue='diagnosis', data=data_sample, color='red')
    plt.savefig("/Users/karola/PycharmProjects/breast_cancer/src/flask-app/static/output_plot_to_display.png",
                bbox_inches='tight', dpi=500)


def split_dataframe(path_to_dataframe):
    df = pd.read_csv(path_to_dataframe)
    important_features = df[['area_se', 'texture_mean', 'texture_worst', 'concave points_worst', 'concavity_worst',
                             'smoothness_worst', 'perimeter_worst', 'smoothness_mean', 'concave points_mean', 'area_worst']]
    return df, important_features


def load_data():
    train = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/train_set.csv")
    test = pd.read_csv("/Users/karola/PycharmProjects/breast_cancer/data/datasets/test_set.csv")
    column_with_labels = 'diagnosis'
    x_train = train.drop([column_with_labels], axis=1)
    x_test = test.drop([column_with_labels], axis=1)
    y_train = train[column_with_labels]
    y_test = test[column_with_labels]
    return x_train, x_test, y_train, y_test


def map_prediction_to_string(prediction):
    return 'malignant' if prediction == 1 else 'benign'


def find_probability(probability):
    return np.round(np.multiply(np.max(probability), 100), 2)


def remove_files_in_dir(directory):
    filelist = [f for f in os.listdir(directory)]
    for f in filelist:
            os.remove(os.path.join(directory, f))


def remove_files_in_dir_png(dir):
    filelist = [f for f in os.listdir(dir) if f == 'output_plot_to_display.png']
    for f in filelist:
        os.remove(os.path.join(dir, f))


if __name__ == '__main__':
    app.run(debug=True, port=5002)
