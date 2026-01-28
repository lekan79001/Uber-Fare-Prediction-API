# Uber Fare Prediction API

This project is a production-ready machine learning API for predicting Uber fares using an XGBoost model. It is containerized with Docker and can be deployed easily.

## Features
- FastAPI backend for real-time predictions
- XGBoost regression model
- Docker and Docker Compose support
- Batch and single prediction endpoints
- Health and model info endpoints

## API Endpoints
- `POST /predict` — Predict fare for a single trip
- `POST /predict-batch` — Predict fares for multiple trips
- `GET /health` — Health check
- `GET /model-info` — Model metadata

## Example Request
```
POST /predict
{
  "pickup_longitude": -73.982,
  "pickup_latitude": 40.767,
  "dropoff_longitude": -73.964,
  "dropoff_latitude": 40.765,
  "pickup_datetime": "2026-01-28T12:30:00"
}
```

## Running Locally
1. Clone the repo:
   ```
   git clone https://github.com/lekan79001/Uber-Fare-Prediction-API.git
   cd Uber-Fare-Prediction-API
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Build and start with Docker Compose:
   ```
   docker-compose up --build
   ```
4. Access the API at [http://localhost:8000/docs](http://localhost:8000/docs)

## requirements.txt
Main dependencies:
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
xgboost==2.0.3
joblib==1.3.2
python-dotenv==1.0.0
python-multipart==0.0.6
prometheus-fastapi-instrumentator==6.1.0
pytest==7.4.4
pytest-cov==4.1.0
httpx==0.26.0
```

## CI/CD with GitHub Actions
This project includes a basic CI workflow to:
- Install dependencies
- Run linting
- Run tests (if present)
- Build Docker image

See `.github/workflows/ci.yml` for details.

---

## License
MIT
