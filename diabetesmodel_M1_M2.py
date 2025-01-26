import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

# Set the tracking URI to a local directory
artifact_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(artifact_dir, exist_ok=True)
mlflow.set_tracking_uri(artifact_dir)

# Create or set an experiment
mlflow.set_experiment("diabetes_experiment")

# Load the versioned dataset
df = pd.read_csv('dataset/diabetes_dataset_v0 copy.csv')

def train_model(n_estimators, max_depth):
    # Split data
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run with n_estimators={n_estimators}, max_depth={max_depth}, mean_squared_error={mse}")

# Run experiments
with mlflow.start_run():
    train_model(n_estimators=10, max_depth=3)

with mlflow.start_run():
    train_model(n_estimators=50, max_depth=5)

with mlflow.start_run():
    train_model(n_estimators=100, max_depth=7)


