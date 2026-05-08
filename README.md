# 🏥 MediScan AI — Multi-Modal Clinical Diagnosis System

> An intelligent, placement-ready AI healthcare platform combining **Natural Language Processing (NLP)** and **Deep Learning (CNN)** for disease prediction from symptoms and chest X-ray images.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-orange) ![Accuracy](https://img.shields.io/badge/X--ray%20Accuracy-97.95%25-brightgreen)

---

## 📌 Project Overview

MediScan AI is a full-stack AI-powered healthcare diagnosis system with two independent pipelines:

| Pipeline | Technology | Output |
|---|---|---|
| 📝 Symptom Analysis | TF-IDF + Random Forest (Calibrated) | Disease name + confidence + top-3 chart |
| 🩻 Radiology Diagnosis | ResNet18 CNN (Fine-tuned) | Normal / Pneumonia |

---

## 🧠 AI/ML Architecture

### Pipeline 1 — Symptom Analysis (NLP + ML)
- **Vectorization:** TF-IDF with unigrams and bigrams (1000 features)
- **Classifier:** Random Forest (200 trees, balanced class weights)
- **Calibration:** Isotonic regression for reliable probability scores
- **Dataset:** 120+ manually crafted samples across 15 diseases
- **Uncertainty guard:** Predictions below 35% confidence flagged as uncertain

### Pipeline 2 — Radiology Diagnosis (Deep Learning)
- **Architecture:** ResNet18 (11M parameters)
- **Pretrained on:** ImageNet
- **Fine-tuned on:** 5,840 chest X-ray images (Kaggle dataset)
- **Validation Accuracy:** 97.95%
- **Classes:** Normal · Pneumonia
- **Safety:** Image quality check + entropy-based uncertainty rejection

---

## 🏥 Supported Diseases — Symptom Model (15 Total)

```
Common Cold      Flu           COVID-19      Pneumonia     Bronchitis
Diabetes         Heart Condition  Hypertension  Anemia     Asthma
Migraine         Dengue        Typhoid       Gastritis     UTI
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend API | FastAPI + Uvicorn | REST API, request handling |
| Frontend UI | Streamlit + Plotly | Interactive web interface, charts |
| ML Model | Scikit-learn | TF-IDF vectorizer + Random Forest |
| DL Model | PyTorch + TorchVision | ResNet18 CNN for image classification |
| Model Saving | Joblib | Save/load .pkl model files |
| Language | Python 3.12 | Core programming language |

---

## 📂 Project Structure

```
AI HEALTHCARE DIAGNOSIS/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app — /symptoms /image /info /health
│   └── requirements.txt         # All dependencies
│
├── frontend/
│   └── app.py                   # Streamlit UI with charts + session history
│
├── ml_models/
│   ├── __init__.py
│   ├── image_model.py           # ResNet18 inference with quality checks
│   ├── image_model.pth          # Saved fine-tuned model weights
│   ├── model_metadata.json      # CV accuracy, class list, model stats
│   ├── symptom_model.pkl        # Saved Random Forest model
│   ├── symptom_model.py         # Symptom inference — top-3 predictions
│   ├── train_image_model.py     # Fine-tune ResNet18 on chest X-ray dataset
│   ├── train_symptom_model.py   # Train ML model (run once)
│   └── vectorizer.pkl           # Saved TF-IDF vectorizer
│
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Train Symptom Model (Run Once)
```bash
cd ml_models
python train_symptom_model.py
```
Output: `✅ Symptom model trained and saved successfully!`

### Step 3 — Train Image Model (Run Once)
```bash
cd ml_models
python train_image_model.py
```
Output: `✅ Training complete! Best val accuracy: 97.95%`

### Step 4 — Start Backend (Keep Terminal Open)
```bash
cd backend

# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m uvicorn main:app --reload

# Mac / Linux
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --reload
```
Backend runs at: `http://127.0.0.1:8000`
API docs at: `http://127.0.0.1:8000/docs`

### Step 5 — Start Frontend (New Terminal, Keep Open)
```bash
cd frontend

# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m streamlit run app.py

# Mac / Linux
streamlit run app.py
```
Frontend runs at: `http://localhost:8501`

> ⚠️ Both Terminal 4 and Terminal 5 must stay running at the same time.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | System info |
| GET | `/health` | Health check — confirms backend is online |
| GET | `/info` | Model metadata, accuracy, class list |
| POST | `/symptoms` | Predict disease from symptom text |
| POST | `/image` | Analyze chest X-ray image |

### Example Request — Symptom Prediction
```bash
curl -X POST http://localhost:8000/symptoms \
  -H "Content-Type: application/json" \
  -d '{"text": "fever cough body ache chills sweating"}'
```

### Example Response
```json
{
  "disease": "Flu",
  "confidence": 0.8732,
  "status": "success",
  "message": "Predicted with 87.3% confidence.",
  "top3": [
    {"disease": "Flu", "confidence": 0.8732},
    {"disease": "COVID-19", "confidence": 0.0721},
    {"disease": "Common Cold", "confidence": 0.0312}
  ]
}
```

---

## 🧪 Sample Symptom Inputs to Test

```
fever cough body ache chills sweating          →  Flu
runny nose sneezing mild fever sore throat     →  Common Cold
loss of smell loss of taste dry cough fever    →  COVID-19
frequent urination excessive thirst fatigue    →  Diabetes
chest pain pressure shortness of breath        →  Heart Condition
throbbing headache nausea light sensitivity    →  Migraine
burning urination frequent urge lower back     →  UTI
high fever severe joint pain rash              →  Dengue
wheezing shortness of breath chest tightness   →  Asthma
prolonged fever stomach pain weakness          →  Typhoid
```

---

## ✨ Key Features

- ✅ **Dual-pipeline AI** — NLP/ML for symptoms + CNN for chest X-rays
- ✅ **97.95% X-ray accuracy** — Fine-tuned ResNet18 on 5,840 images
- ✅ **15 disease prediction** — Symptom model with 120+ training samples
- ✅ **Confidence bar charts** — Plotly charts show top-3 probabilities
- ✅ **Session history** — All predictions stored in sidebar during session
- ✅ **Image safety checks** — Quality validation + uncertainty rejection
- ✅ **Full REST API** — FastAPI backend with Swagger docs at `/docs`
- ✅ **Model info endpoint** — `/info` shows accuracy, class list, model stats
- ✅ **Auto class detection** — Image model auto-detects output classes from weights

---

## 📊 Model Performance

| Model | Type | Dataset | Accuracy |
|---|---|---|---|
| Symptom Model | Random Forest (Calibrated) | 120 samples, 15 classes | ~85% CV |
| Image Model | ResNet18 (Fine-tuned) | 5,840 chest X-rays | **97.95%** |

---

## 🎓 Academic & Placement Value

- Demonstrates real ML + DL integration in one production-style system
- Uses industry-standard architectures (ResNet18, Random Forest, TF-IDF)
- Full-stack deployment — FastAPI backend + Streamlit frontend
- REST API design with proper error handling and structured JSON responses
- Calibrated probabilities — not just argmax, real confidence scores
- Safety-oriented AI — uncertainty rejection, image quality checks
- Suitable for Final Year Project / Mega Project / Placement Portfolio

---

## ⚠️ Disclaimer

This system is intended for **academic and research purposes only**.
It is not a certified medical device and must not be used for real-world clinical diagnosis.
Always consult a qualified healthcare professional.

---

## 👤 Author

**HasanAli**
