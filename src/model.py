# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Train model with dummy data
data = {
    'user_total_bookings': [5, 10, 3, 7, 2, 1, 12, 4],
    'user_cancel_rate': [0.2, 0.5, 0.0, 0.1, 0.3, 0.0, 0.4, 0.25],
    'price': [100, 500, 50, 200, 150, 80, 600, 120],
    'duration_days': [3, 7, 1, 4, 2, 1, 5, 3],
    'days_before_checkin': [15, 2, 20, 10, 1, 30, 3, 7],
    'payment_completed': [1, 0, 1, 1, 0, 1, 1, 1],
    'cancelled': [0, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and target variable
X = df.drop('cancelled', axis=1)
y = df['cancelled']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
model_path = os.path.join(os.path.dirname(__file__), 'booking_cancel_model.joblib')
joblib.dump(model, model_path)

print("✅ Model trained and saved successfully at:", model_path)
