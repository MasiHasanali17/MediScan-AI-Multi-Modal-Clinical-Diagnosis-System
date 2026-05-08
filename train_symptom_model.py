import pandas as pd
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# ─── Dataset: 15 diseases, 120+ samples ───────────────────────────────────────
data = {
    "symptoms": [
        # Common Cold (10)
        "runny nose sneezing mild fever sore throat",
        "blocked nose cold sniffles watery eyes",
        "sneezing cough runny nose congestion",
        "mild fever sore throat nasal congestion",
        "runny nose cough low grade fever",
        "nasal discharge sneezing throat irritation",
        "stuffy nose mild headache sneezing",
        "cold runny nose mild body ache",
        "sore throat sneezing mild fatigue",
        "congestion watery eyes sneezing cough",

        # Flu (10)
        "high fever body ache chills fatigue",
        "sudden fever headache muscle pain chills",
        "severe fatigue high fever sore throat",
        "body pain high fever sweating chills",
        "shivering high temperature body ache cough",
        "fever headache severe muscle ache fatigue",
        "chills sweating high fever joint pain",
        "sudden onset fever muscle weakness fatigue",
        "high fever dry cough body pain",
        "intense fatigue fever headache body ache",

        # COVID-19 (10)
        "loss of smell loss of taste fever cough",
        "dry cough fever fatigue breathing difficulty",
        "loss of taste shortness of breath fever",
        "persistent cough fever chills loss of smell",
        "fever difficulty breathing dry cough",
        "loss of smell fever fatigue body ache",
        "shortness of breath chest pain dry cough fever",
        "loss of taste loss of smell cough fatigue",
        "fever sore throat dry cough oxygen low",
        "breathlessness fever chills loss of taste",

        # Pneumonia (8)
        "chest pain cough fever difficulty breathing",
        "productive cough fever chills breathing difficulty",
        "chest tightness fever cough phlegm",
        "wet cough fever rapid breathing fatigue",
        "chest congestion high fever breathing trouble",
        "cough with mucus fever chest pain",
        "shortness of breath fever cough sweating",
        "chest infection cough high fever tiredness",

        # Bronchitis (7)
        "persistent cough mucus low fever chest discomfort",
        "cough phlegm mild fever fatigue",
        "chronic cough chest tightness mucus",
        "cough with yellow mucus mild fever",
        "wheezing cough chest discomfort mild temperature",
        "persistent wet cough fatigue mild fever",
        "cough phlegm shortness of breath",

        # Diabetes (9)
        "frequent urination excessive thirst fatigue",
        "increased hunger frequent urination weight loss",
        "blurry vision fatigue frequent urination thirst",
        "slow wound healing fatigue frequent urination",
        "excessive thirst urination tingling feet",
        "weight loss increased appetite frequent urination",
        "fatigue thirst frequent urination blurred vision",
        "numbness in hands feet excessive thirst",
        "frequent urination unexplained weight loss fatigue",

        # Heart Condition (8)
        "chest pain pressure shortness of breath",
        "chest tightness pain radiating arm sweating",
        "heart palpitations chest discomfort breathlessness",
        "chest pain jaw pain left arm numbness",
        "shortness of breath chest pressure sweating",
        "rapid heartbeat chest pain dizziness",
        "chest pain fatigue swollen ankles breathlessness",
        "heart racing chest tightness dizziness sweating",

        # Hypertension (7)
        "severe headache dizziness blurred vision",
        "headache neck pain dizziness",
        "persistent headache nosebleed dizziness",
        "blurred vision headache fatigue chest tightness",
        "morning headache vision changes fatigue",
        "dizziness headache shortness of breath",
        "neck stiffness headache fatigue palpitations",

        # Anemia (7)
        "fatigue pale skin dizziness weakness",
        "extreme tiredness pale gums weakness",
        "shortness of breath fatigue pale skin cold hands",
        "dizziness weakness fatigue cold extremities",
        "fatigue brittle nails pale skin headache",
        "weakness tiredness cold hands feet paleness",
        "dizziness headache fatigue pale conjunctiva",

        # Asthma (7)
        "wheezing shortness of breath chest tightness",
        "difficulty breathing wheezing cough night",
        "chest tightness wheezing breathlessness exercise",
        "wheezing coughing breathlessness triggers",
        "shortness of breath cough wheeze allergy",
        "breathing difficulty chest tightness wheezing night",
        "wheezing cough shortness of breath cold air",

        # Migraine (8)
        "throbbing headache nausea light sensitivity",
        "severe headache one side nausea vomiting",
        "pulsating head pain light sound sensitivity",
        "migraine aura visual disturbance headache",
        "severe throbbing headache nausea sensitivity",
        "head pain nausea vomiting light sensitivity",
        "one sided headache throbbing nausea",
        "headache with aura nausea vomiting",

        # Dengue (8)
        "high fever severe joint pain rash",
        "dengue fever bone pain rash fatigue",
        "sudden high fever severe body ache rash",
        "joint pain muscle pain fever rash",
        "high fever rash headache eye pain",
        "fever rash joint pain platelet low",
        "severe body pain fever skin rash",
        "eye pain fever severe joint pain rash",

        # Typhoid (7)
        "prolonged fever stomach pain weakness",
        "high fever abdominal pain constipation",
        "continuous fever headache stomach ache",
        "fever rose spots abdominal pain fatigue",
        "sustained fever nausea stomach discomfort",
        "prolonged high fever loss of appetite fatigue",
        "fever abdominal cramps weakness nausea",

        # Gastritis (7)
        "stomach pain bloating nausea after eating",
        "abdominal discomfort nausea vomiting",
        "upper stomach pain burning indigestion",
        "nausea bloating stomach cramps",
        "stomach burning acid reflux nausea",
        "upper abdominal pain loss of appetite nausea",
        "indigestion bloating stomach ache",

        # UTI (7)
        "burning urination frequent urge lower back pain",
        "painful urination cloudy urine frequent urge",
        "frequent urination burning sensation pelvic pain",
        "urinary urgency burning lower abdominal pain",
        "cloudy urine burning urination back pain",
        "frequent urge urinate painful urination fever",
        "lower back pain burning urination frequent urge",
    ],
    "disease": [
        *["Common Cold"] * 10,
        *["Flu"] * 10,
        *["COVID-19"] * 10,
        *["Pneumonia"] * 8,
        *["Bronchitis"] * 7,
        *["Diabetes"] * 9,
        *["Heart Condition"] * 8,
        *["Hypertension"] * 7,
        *["Anemia"] * 7,
        *["Asthma"] * 7,
        *["Migraine"] * 8,
        *["Dengue"] * 8,
        *["Typhoid"] * 7,
        *["Gastritis"] * 7,
        *["UTI"] * 7,
    ]
}

df = pd.DataFrame(data)
print(f"Dataset: {len(df)} samples, {df['disease'].nunique()} diseases")
print(df['disease'].value_counts())

# ─── Vectorizer ────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=1,
    max_features=1000
)

X = vectorizer.fit_transform(df["symptoms"])
y = df["disease"]

# ─── Model: Random Forest (robust, calibrated probabilities) ───────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)

# Cross-validation to show accuracy
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"\nCross-validation accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")

# Calibrate probabilities for reliable confidence scores
model = CalibratedClassifierCV(rf, cv=5, method="isotonic")
model.fit(X, y)

# ─── Save models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
joblib.dump(model, os.path.join(BASE, "symptom_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE, "vectorizer.pkl"))

# Save metadata for /info endpoint
import json
metadata = {
    "model_type": "Random Forest (Calibrated)",
    "n_estimators": 200,
    "n_classes": int(df['disease'].nunique()),
    "classes": sorted(df['disease'].unique().tolist()),
    "n_training_samples": len(df),
    "cv_accuracy": round(float(scores.mean()), 4),
    "vectorizer": "TF-IDF (unigrams + bigrams)",
    "features": int(X.shape[1])
}
with open(os.path.join(BASE, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Symptom model trained and saved successfully!")
print(f"   Classes: {metadata['classes']}")