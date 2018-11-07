import pandas as pd
import numpy as np
import os
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
    sample = preprocess_sample(path_to_sample="/Users/karola/PycharmProjects/breast_cancer/src/flask-app/uploaded/sample.csv")
    prediction, probability = predictor_log_reg(sample)
    prediction = map_prediction_to_string(prediction)
    probability = find_probability(probability)
    return render_template('prediction.html', prediction=prediction, probability=probability)


def preprocess_sample(path_to_sample):
    sample = pd.read_csv(path_to_sample)
    new_sample = np.ravel(sample.values[0, :])
    test_sample = new_sample.reshape(1, -1)
    return test_sample


def predictor_log_reg(sample):
    x_train, x_test, y_train, y_test = load_data()
    logistic_regression = LogisticRegression()
    fit_log_reg = logistic_regression.fit(x_train, y_train)
    # sample = preprocess_sample(path_to_sample)
    prediction = fit_log_reg.predict(sample)
    probability = fit_log_reg.predict_proba(sample)
    return prediction, probability


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
    return np.multiply(np.max(probability),100)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
