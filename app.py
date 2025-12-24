from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Initialize the App
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time inference API for detecting fraudulent transactions",
    version="1.0.0"
)

# Load the trained model
try:
    model = joblib.load("fraud_model.joblib")
except:
    print("Model not found. Please run train_model.py first.")

# Define the Input Schema (Data Validation using Pydantic)
class Transaction(BaseModel):
    amount: float
    time_delta: float
    merchant_score: float
    location_score: float
    device_score: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running. Go to /docs for the UI."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # 1. Convert input data to DataFrame (matching training format)
    input_data = pd.DataFrame([transaction.dict().values()], columns=transaction.dict().keys())
    
    # 2. Make Prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Probability of being fraud
        
        # 3. Return Result
        result = "FRAUD" if prediction == 1 else "LEGITIMATE"
        return {
            "prediction": result,
            "fraud_probability": round(float(probability), 4),
            "status": 200
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running with python directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)