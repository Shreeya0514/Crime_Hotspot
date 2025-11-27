from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Load trained model
model = joblib.load("crime_model.pkl")

class CrimeInput(BaseModel):
    area: float
    population: int
    month: int
    police_stations: int

@app.post("/predict")
def predict_hotspot(data: CrimeInput):
    features = np.array([[data.area, data.population, data.month, data.police_stations]])
    prediction = model.predict(features)
    return {"hotspot": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
