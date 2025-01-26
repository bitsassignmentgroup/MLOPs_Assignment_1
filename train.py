import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
import os


# Load the Diabetes dataset
df = pd.read_csv('dataset/diabetes_dataset_v0 copy.csv')
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 7]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best hyperparameters: ", best_params)

# Train the best model
best_model = grid_search.best_estimator_

# Save the best model using joblib
joblib.dump(best_model, 'best_model.pkl')

# Evaluate the best model
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Log parameters, metrics, and model with MLflow
artifact_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(artifact_dir, exist_ok=True)
mlflow.set_tracking_uri(artifact_dir)
mlflow.set_experiment("diabetes_experiment")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.sklearn.log_model(best_model, "model")
