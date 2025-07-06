import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# current file এর directory থেকে মডেল ফাইলের সঠিক path তৈরি করলাম
model_path = os.path.join(os.path.dirname(__file__), 'booking_cancel_model.joblib')
print(f"Loading model from: {model_path}")

model = joblib.load(model_path)

class Features(BaseModel):
    user_total_bookings: int
    user_cancel_rate: float
    price: float
    duration_days: int
    days_before_checkin: int
    payment_completed: int

@app.post("/predict")
def predict(features: Features):
    input_data = np.array([
        features.user_total_bookings,
        features.user_cancel_rate,
        features.price,
        features.duration_days,
        features.days_before_checkin,
        features.payment_completed,
    ]).reshape(1, -1)
    
    probability = model.predict_proba(input_data)[0][1]
    return {"cancel_probability": probability}
