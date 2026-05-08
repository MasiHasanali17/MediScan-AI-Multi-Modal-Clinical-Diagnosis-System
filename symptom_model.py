import joblib
import os
import json
import numpy as np

BASE = os.path.dirname(__file__)

# Load model and vectorizer at startup
model = joblib.load(os.path.join(BASE, "symptom_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE, "vectorizer.pkl"))

# Load metadata if available
_metadata_path = os.path.join(BASE, "model_metadata.json")
_metadata = {}
if os.path.exists(_metadata_path):
    with open(_metadata_path) as f:
        _metadata = json.load(f)


def predict_symptoms(text: str) -> dict:
    """
    Predict disease from symptom text.
    Returns top prediction + top 3 probabilities for visualization.
    """
    text = text.strip().lower()

    if len(text) < 3:
        return {
            "disease": "Invalid Input",
            "confidence": 0.0,
            "status": "error",
            "message": "Please describe your symptoms in more detail.",
            "top3": []
        }

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    # Top prediction
    top_idx = int(np.argmax(probs))
    top_disease = classes[top_idx]
    top_conf = float(probs[top_idx])

    # Top 3 for chart
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"disease": classes[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    # Uncertainty guard
    if top_conf < 0.35:
        return {
            "disease": "Uncertain",
            "confidence": round(top_conf, 4),
            "status": "uncertain",
            "message": "Symptoms too vague or mixed. Please add more specific symptoms.",
            "top3": top3
        }

    return {
        "disease": top_disease,
        "confidence": round(top_conf, 4),
        "status": "success",
        "message": f"Predicted with {round(top_conf * 100, 1)}% confidence.",
        "top3": top3
    }


def get_model_info() -> dict:
    """Return model metadata for the /info endpoint."""
    return {
        "symptom_model": _metadata if _metadata else {
            "model_type": "Random Forest (Calibrated)",
            "classes": list(model.classes_),
            "n_classes": len(model.classes_)
        },
        "image_model": {
            "architecture": "ResNet18",
            "pretrained": "ImageNet",
            "classes": ["Normal", "Pneumonia", "COVID-19"],
            "input_size": "224x224"
        }
    }