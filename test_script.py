import pytest
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dataset
df = pd.read_csv('dataset/diabetes_dataset_v0 copy.csv')

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
    
