import pytest
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dataset
df = pd.read_csv('dataset/diabetes_dataset_v0.csv')

# Function to train the model
def train_model(n_estimators, max_depth):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.sklearn.log_model(model, "model")
    return mse


def test_dataset_integrity():
    assert not df.empty, "The dataset should not be empty."
    assert 'target' in df.columns, "The dataset is missing the required column: 'target'"

def test_model_training():
    n_estimators = 10
    max_depth = 3
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    assert len(predictions) == len(X_test), "The number of predictions should match the number of test samples."
    assert mse >= 0, "Mean Squared Error should be non-negative."

def test_mlflow_logging():
    n_estimators = 10
    max_depth = 3
    with mlflow.start_run() as run:
        mse = train_model(n_estimators, max_depth)
        run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id).data
    assert run_data.params["n_estimators"] == str(n_estimators), "n_estimators parameter was not logged correctly."
    assert run_data.params["max_depth"] == str(max_depth), "max_depth parameter was not logged correctly."
    assert "mean_squared_error" in run_data.metrics, "Mean Squared Error metric was not logged."
    artifacts = client.list_artifacts(run_id, "model")
    assert len(artifacts) > 0, "Model artifact was not logged."
