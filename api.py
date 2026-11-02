from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "house_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API")


class HouseFeatures(BaseModel):
    area_sqft: int = Field(gt=0)
    bedrooms: int = Field(ge=1)
    bathrooms: int = Field(ge=1)
    house_age: int = Field(ge=0)


@app.post("/predict")
def predict_price(house: HouseFeatures):
    input_data = pd.DataFrame([house.model_dump()])
    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }