from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
import time
import json
from typing import Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Uber Fare Prediction API",
    description="XGBoost-based Uber fare prediction service",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_metadata = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, model_metadata
    try:
        logger.info("Loading XGBoost model...")
        model = joblib.load('xgb_model.joblib')
        
        # Load metadata
        with open('xgb_model_documentation.json', 'r') as f:
            model_metadata = json.load(f)
        
        logger.info(f"Model loaded successfully! Version: {model_metadata.get('version', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Request/Response models
class PredictionRequest(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    pickup_datetime: str  # ISO format string
    
    class Config:
        json_schema_extra = {
            "example": {
                "pickup_longitude": -73.982,
                "pickup_latitude": 40.767,
                "dropoff_longitude": -73.964,
                "dropoff_latitude": 40.765,
                "pickup_datetime": "2026-01-28T12:30:00"
            }
        }

class PredictionResponse(BaseModel):
    fare_amount: float
    model_version: str
    timestamp: float

@app.get("/")
async def root():
    return {
        "message": "Uber Fare Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ready",
        "model_version": model_metadata.get('version', 'unknown') if model_metadata else 'unknown'
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return model_metadata

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict Uber fare"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Calculate trip_distance using Haversine formula (miles)
        def haversine(lon1, lat1, lon2, lat2):
            R = 3959  # Earth radius in miles
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        trip_distance = haversine(
            request.pickup_longitude,
            request.pickup_latitude,
            request.dropoff_longitude,
            request.dropoff_latitude
        )

        # Parse pickup_datetime
        pickup_dt = pd.to_datetime(request.pickup_datetime)
        pickup_hour = pickup_dt.hour

        # Rush hour: 7-9am or 4-7pm
        is_rush_hour = 1 if (7 <= pickup_hour <= 9 or 16 <= pickup_hour <= 19) else 0

        # Manhattan bounding box
        pickup_in_manhattan = 1 if (40.70 <= request.pickup_latitude <= 40.88 and -74.02 <= request.pickup_longitude <= -73.93) else 0

        distance_rush_interaction = trip_distance * is_rush_hour

        features = pd.DataFrame([{
            'trip_distance': trip_distance,
            'pickup_hour': pickup_hour,
            'is_rush_hour': is_rush_hour,
            'pickup_in_manhattan': pickup_in_manhattan,
            'distance_rush_interaction': distance_rush_interaction
        }])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        logger.info(f"Prediction made: ${prediction:.2f}")
        
        return PredictionResponse(
            fare_amount=float(prediction),
            model_version=model_metadata.get('version', '1.0.0') if model_metadata else '1.0.0',
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction endpoint"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        def haversine(lon1, lat1, lon2, lat2):
            R = 3959  # Earth radius in miles
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        for req in requests:
            trip_distance = haversine(
                req.pickup_longitude,
                req.pickup_latitude,
                req.dropoff_longitude,
                req.dropoff_latitude
            )
            pickup_dt = pd.to_datetime(req.pickup_datetime)
            pickup_hour = pickup_dt.hour
            is_rush_hour = 1 if (7 <= pickup_hour <= 9 or 16 <= pickup_hour <= 19) else 0
            pickup_in_manhattan = 1 if (40.70 <= req.pickup_latitude <= 40.88 and -74.02 <= req.pickup_longitude <= -73.93) else 0
            distance_rush_interaction = trip_distance * is_rush_hour
            features = pd.DataFrame([{
                'trip_distance': trip_distance,
                'pickup_hour': pickup_hour,
                'is_rush_hour': is_rush_hour,
                'pickup_in_manhattan': pickup_in_manhattan,
                'distance_rush_interaction': distance_rush_interaction
            }])
            prediction = model.predict(features)[0]
            predictions.append({
                "fare_amount": float(prediction),
                "request": req.dict()
            })

        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))