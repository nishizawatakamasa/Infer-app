import joblib
import numpy as np
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sklearn.linear_model import LinearRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://infer-app-front.onrender.com'],
    allow_credentials=False,
    allow_methods=['GET'],
    allow_headers=['*'],
)

MODEL_FILE = 'models/model.joblib'
model: LinearRegression = joblib.load(MODEL_FILE)

class PredictionQuery(BaseModel):
    age: int
    height: int

    @field_validator('age', 'height')
    @classmethod
    def feature_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError('feature must be positive')
        return value

class PredictionResponse(BaseModel):
    weight: float

@app.get('/predict', response_model=PredictionResponse)
def predict(query: PredictionQuery = Depends()) -> PredictionResponse:
    '''Predicts a weight based on input features (age and height) provided as query parameters.'''
    weight = model.predict(np.array([[query.age, query.height]]))[0]
    return PredictionResponse(weight=weight)
