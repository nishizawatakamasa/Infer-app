import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from sklearn.linear_model import LinearRegression

MODEL_FILE = 'models/model.joblib'

app = FastAPI()

try:
    model: LinearRegression = joblib.load(MODEL_FILE)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail=f"Model file not found: {MODEL_FILE}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model from {MODEL_FILE}: {e}")

class InputData(BaseModel):
    value: float

    @field_validator('value', mode='before')
    @classmethod
    def value_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError('Value must be positive')
        return value

class PredictionResult(BaseModel):
    value: float
    pred: float

@app.post('/predict/', response_model=PredictionResult)
def predict(data: InputData) -> PredictionResult:
    '''This endpoint takes a single value as input and returns the model's prediction.'''
    value = data.value
    try:
        pred = model.predict(np.array([[value]]))[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictionResult(value=value, pred=pred)
