
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set MLflow experiment
mlflow.set_experiment("House Price Prediction")

# Load dataset
df = pd.read_csv("data/house_prices.csv")

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("training_records", len(X_train))
    mlflow.log_param("features", list(X.columns))

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log trained model
    mlflow.sklearn.log_model(model, name="house_price_model")

    # Display results
    print("Model Evaluation Results")
    print("------------------------")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# Save model using Joblib
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_model.joblib")

print("\nModel saved successfully!")
print("MLflow tracking completed!")
