import os

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score


TEST_MODEL_NAME = os.getenv('TEST_MODEL_NAME')
TEST_MODEL_VERSION = os.getenv('TEST_MODEL_VERSION')
TEST_TEST_SET = os.getenv('TEST_TEST_SET')


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{TEST_MODEL_NAME}/{TEST_MODEL_VERSION}"
)


def test_simple_input():
    inputs = ['Bonjour, je suis un test']
    prediction = mlflow_model.predict(inputs)
    assert prediction[0] in [0, 1]


def test_unusual_inputs():
    inputs = ['&$@é', '']
    prediction = mlflow_model.predict(inputs)
    assert all([x in [0, 1] for x in prediction])


def test_obvious_inputs():
    inputs = [
        'Ce film est génial, très bien réalisé et les acteurs excellents',
        'c nul']
    prediction = mlflow_model.predict(inputs)
    assert prediction[0] == 1
    assert prediction[1] == 0


def test_accuracy():
    df = pd.read_csv(TEST_TEST_SET, index_col=0)
    X_test = df['review']
    y_test = df['polarity']
    y_pred = mlflow_model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.9
