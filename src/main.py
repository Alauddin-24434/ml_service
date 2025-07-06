import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Construct the correct path to the model file (assuming it's in the same folder)
model_path = os.path.join(os.path.dirname(__file__), 'booking_cancel_model.joblib')
print(f"Loading model from: {model_path}")

# Load the ML model
model = joblib.load(model_path)

# Define input data schema using Pydantic for validation
class Features(BaseModel):
    user_total_bookings: int
    user_cancel_rate: float
    price: float
    duration_days: int
    days_before_checkin: int
    payment_completed: int

# Root endpoint to check if the service is running
@app.get("/")
def root():
    return {"message": "ML service is up and running successfully!"}

# Prediction endpoint
@app.post("/predict")
def predict(features: Features):
    # Convert input features to numpy array with correct shape
    input_data = np.array([
        features.user_total_bookings,
        features.user_cancel_rate,
        features.price,
        features.duration_days,
        features.days_before_checkin,
        features.payment_completed,
    ]).reshape(1, -1)
    
    # Get the probability of cancellation from the model
    probability = model.predict_proba(input_data)[0][1]
    
    # Return the probability as JSON response
    return {"cancel_probability": probability}
