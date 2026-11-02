# House Price Prediction Using MLOps

## 1. Project Overview

This project implements a machine learning pipeline to predict house prices using Linear Regression.

It demonstrates the basic MLOps workflow, including:

- Synthetic dataset generation
- Data exploration and preprocessing
- Model training and evaluation
- Experiment tracking using MLflow
- Model serialization using Joblib
- Model deployment using FastAPI
- Source code version control using Git

## 2. Project Structure

```text
house-price-mlops/
│
├── data/
│   └── house_prices.csv
│
├── models/
│   └── house_model.joblib
│
├── generate_data.py
├── data_exploration.py
├── train.py
├── api.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── mlflow.db
```

## 3. Dataset Description

The project uses a synthetic dataset containing 500 house records.

| Feature | Description |
|---|---|
| area_sqft | House area in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| house_age | Age of the house in years |
| price | House price (target variable) |

The dataset is generated using a predefined pricing formula with random noise.

## 4. Technologies and Libraries Used

### Python Libraries

| Library | Purpose in this project |
|---|---|
| **NumPy** | Generates random values and adds normally distributed noise while creating synthetic data. |
| **Pandas** | Creates and manages datasets, reads/writes CSV files, handles features and target variables, and prepares input data for prediction. |
| **Scikit-learn** | Provides `train_test_split` for splitting data, `LinearRegression` for model training, and evaluation metrics such as MAE, RMSE and R². |
| **MLflow** | Tracks machine learning experiments, parameters, evaluation metrics and trained model artifacts. |
| **Joblib** | Saves the trained model to a file and loads it later for prediction without retraining. |
| **FastAPI** | Creates REST API endpoints to expose the trained machine learning model for external applications. |
| **Pydantic** | Validates incoming API data and ensures that house features have valid types and values. |
| **Uvicorn** | Runs the FastAPI application as an ASGI web server. |

### Python Standard Library

| Module | Purpose |
|---|---|
| **os** | Creates directories when required, using functions such as `os.makedirs()`. |
| **pathlib** | Handles file and directory paths in a platform-independent way. |

### Development Tools

| Tool | Purpose |
|---|---|
| **Git** | Tracks source-code changes and maintains project history. |
| **PyCharm** | IDE used for writing, running and debugging Python code. |
| **Swagger UI** | Provides an interactive interface to test API endpoints. |

## 5. Installation and Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Generate Dataset

Run:

```bash
python generate_data.py
```

The dataset will be generated at:

`data/house_prices.csv`

## 7. Data Exploration

Run:

```bash
python data_exploration.py
```

This script displays sample records, dataset information, statistical summaries, missing-value checks and duplicate records.

## 8. Model Training

Run:

```bash
python train.py
```

The training pipeline:

1. Loads the dataset using Pandas.
2. Separates input features (X) and target variable (y).
3. Splits the data into 80% training and 20% testing sets.
4. Trains a Linear Regression model.
5. Evaluates model performance.
6. Saves the trained model using Joblib.
7. Logs parameters, metrics and model artifacts in MLflow.

The trained model is saved at:

`models/house_model.joblib`

## 9. Model Evaluation

The model achieved the following results:

| Metric | Result |
|---|---:|
| MAE | 403,052.88 |
| RMSE | 515,307.41 |
| R² Score | 0.9548 |

These results are based on the synthetic dataset used in this project.

## 10. MLflow Experiment Tracking

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

http://127.0.0.1:5000

Experiment name: `House Price Prediction`

MLflow tracks model parameters, evaluation metrics and model artifacts, allowing experiments to be reviewed and compared.

## 11. FastAPI Deployment

Start the API server:

```bash
uvicorn api:app --reload
```

API base URL:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

### Prediction Endpoint

**POST** `/predict`

Example request:

```json
{
  "area_sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "house_age": 5
}
```

Example response:

```json
{
  "predicted_price": 7289653.7
}
```

## 12. Version Control Using Git

Git is used to maintain project source code and track changes.

Stage changes:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Add project documentation"
```

## 13. Limitations

- The dataset contains synthetic rather than real-world housing data.
- The model uses only four input features.
- Performance on synthetic data does not establish accuracy on real housing prices.
- The API is intended as a demonstration and does not include production authentication or monitoring.


