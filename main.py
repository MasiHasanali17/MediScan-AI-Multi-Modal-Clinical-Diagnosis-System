import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.symptom_model import predict_symptoms, get_model_info
from ml_models.image_model import MedicalImageAI

app = FastAPI(
    title="Nexus Health AI",
    description="AI-Driven Multi-Modal Healthcare Diagnosis System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

image_ai = MedicalImageAI()


class SymptomInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "status": "Nexus Health AI Online",
        "version": "2.0.0",
        "endpoints": ["/symptoms", "/image", "/info", "/health"]
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}


@app.get("/info")
def model_info():
    return get_model_info()


@app.post("/symptoms")
def predict_symptoms_api(data: SymptomInput):
    try:
        if not data.text or len(data.text.strip()) < 3:
            raise HTTPException(status_code=400, detail="Please enter at least one symptom.")
        result = predict_symptoms(data.text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/image")
async def predict_image_api(file: UploadFile = File(...)):
    try:
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        result = image_ai.predict(image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")