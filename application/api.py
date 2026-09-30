from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np
import time
from application.rag_service import rag_client

app = FastAPI(
    title="Nexus RCA Inference API",
    description="Backend API for running Root Cause Analysis inference.",
    version="1.0.0"
)

# Placeholder for MLflow integration
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("nexus_rca")

class IncidentRequest(BaseModel):
    incident_id: str
    telemetry_window_s: int = 30
    services_involved: list[str]

class PredictionResponse(BaseModel):
    incident_id: str
    top_predictions: list[dict]
    evidence: list[str]
    processing_time_ms: float
    rag_summary: str = ""

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_rca(request: IncidentRequest):
    start_time = time.time()
    
    # In a real scenario, we would:
    # 1. Fetch telemetry from a Feature Store (Redis/Feast)
    # 2. Extract features using models/gru_rca.py
    # 3. Predict using the loaded MLflow model
    
    # For now, we mock the inference engine based on the previous Streamlit logic
    if "currency" in request.incident_id:
        preds = [{"service": "currencyservice", "probability": 89.4}, 
                 {"service": "checkoutservice", "probability": 8.1}, 
                 {"service": "paymentservice", "probability": 2.5}]
        evidence = ["Spike in currencyservice memory usage (+45%) at T-60s", "Downstream latency propagation to checkoutservice"]
    elif "recommendation" in request.incident_id:
        preds = [{"service": "recommendationservice", "probability": 94.2}, 
                 {"service": "frontend", "probability": 4.1}, 
                 {"service": "productcatalogservice", "probability": 1.7}]
        evidence = ["recommendationservice disk I/O saturated at 100%", "Frontend p90 latency increased by 2000ms"]
    else:
        preds = [{"service": "emailservice", "probability": 78.5}, 
                 {"service": "checkoutservice", "probability": 15.2}, 
                 {"service": "cartservice", "probability": 6.3}]
        evidence = ["TCP retransmission rate increased in emailservice", "Checkout flow stalled waiting for email confirmation"]
        
    processing_time = (time.time() - start_time) * 1000
    
    # Use RAG service to generate a summary based on the top prediction
    top_service = preds[0]["service"]
    rag_summary = rag_client.generate_rca_summary(top_service, evidence)
    
    # Log inference request to MLflow (as a basic MLOps setup)
    # with mlflow.start_run():
    #    mlflow.log_param("incident_id", request.incident_id)
    #    mlflow.log_metric("processing_time_ms", processing_time)
    
    return PredictionResponse(
        incident_id=request.incident_id,
        top_predictions=preds,
        evidence=evidence,
        processing_time_ms=processing_time,
        rag_summary=rag_summary
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
