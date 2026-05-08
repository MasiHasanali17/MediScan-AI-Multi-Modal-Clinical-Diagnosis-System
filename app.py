import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus Health AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND = "http://localhost:8000"

# ─── Session state ─────────────────────────────────────────────────────────────
if "symptom_history" not in st.session_state:
    st.session_state.symptom_history = []
if "image_history" not in st.session_state:
    st.session_state.image_history = []

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Nexus Health AI")
    st.caption("AI-Driven Multi-Modal Diagnosis System")
    st.divider()

    # Backend status
    try:
        r = requests.get(f"{BACKEND}/health", timeout=2)
        if r.status_code == 200:
            st.success("Backend: Online ✓")
        else:
            st.error("Backend: Error")
    except Exception:
        st.error("Backend: Offline — start uvicorn first")

    # Model info
    with st.expander("Model Info"):
        try:
            info = requests.get(f"{BACKEND}/info", timeout=2).json()
            sm = info.get("symptom_model", {})
            im = info.get("image_model", {})
            st.markdown(f"**Symptom Model**")
            st.markdown(f"- Type: `{sm.get('model_type', 'N/A')}`")
            st.markdown(f"- Diseases: `{sm.get('n_classes', 'N/A')}`")
            st.markdown(f"- Samples: `{sm.get('n_training_samples', 'N/A')}`")
            if sm.get('cv_accuracy'):
                st.markdown(f"- CV Accuracy: `{sm['cv_accuracy']*100:.1f}%`")
            st.markdown(f"**Image Model**")
            st.markdown(f"- Architecture: `{im.get('architecture', 'N/A')}`")
            st.markdown(f"- Classes: `{', '.join(im.get('classes', []))}`")
        except Exception:
            st.info("Start backend to see model info.")

    st.divider()
    st.subheader("Session History")

    if st.session_state.symptom_history:
        st.markdown("**Symptom Predictions**")
        for h in reversed(st.session_state.symptom_history[-5:]):
            st.markdown(f"- `{h['disease']}` ({h['confidence']}%) — {h['time']}")
    else:
        st.caption("No symptom predictions yet.")

    if st.session_state.image_history:
        st.markdown("**Image Diagnoses**")
        for h in reversed(st.session_state.image_history[-5:]):
            st.markdown(f"- `{h['prediction']}` ({h['confidence']}%) — {h['time']}")
    else:
        st.caption("No image diagnoses yet.")

    if st.button("Clear History"):
        st.session_state.symptom_history = []
        st.session_state.image_history = []
        st.rerun()

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 AI Multi-Modal Healthcare Diagnosis Platform")
st.markdown("Combining **Natural Language Processing** and **Deep Learning** for intelligent disease prediction.")
st.divider()

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝  Symptom Analysis (NLP + ML)", "🩻  Radiology AI (Deep Learning)"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Symptom Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Describe Your Symptoms")
        st.caption("Type symptoms in plain English — the model uses TF-IDF + Random Forest to predict the most likely disease.")

        sample_symptoms = [
            "Select a sample...",
            "fever cough cold sniffles runny nose",
            "high fever body ache chills sweating",
            "loss of smell loss of taste dry cough fever",
            "frequent urination excessive thirst fatigue",
            "chest pain pressure shortness of breath sweating",
            "throbbing headache nausea light sensitivity",
            "burning urination frequent urge lower back pain",
            "wheezing shortness of breath chest tightness",
            "high fever severe joint pain rash",
        ]

        selected = st.selectbox("Or pick a sample input:", sample_symptoms)
        symptom_text = st.text_area(
            "Enter symptoms:",
            value="" if selected == "Select a sample..." else selected,
            height=100,
            placeholder="e.g. fever cough body ache chills fatigue"
        )

        predict_btn = st.button("🔍 Predict Disease", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if predict_btn:
            if not symptom_text.strip():
                st.warning("Please enter some symptoms first.")
            else:
                with st.spinner("Analyzing symptoms..."):
                    try:
                        res = requests.post(
                            f"{BACKEND}/symptoms",
                            json={"text": symptom_text},
                            timeout=10
                        ).json()

                        status = res.get("status", "")
                        disease = res.get("disease", "Unknown")
                        confidence = res.get("confidence", 0.0)
                        top3 = res.get("top3", [])

                        if status == "error":
                            st.error(f"⚠️ {res.get('message', 'Invalid input')}")

                        elif status == "uncertain":
                            st.warning(f"🤔 **Uncertain** — {res.get('message', '')}")
                            if top3:
                                st.caption("Closest matches:")
                                for item in top3:
                                    st.markdown(f"- {item['disease']}: `{item['confidence']*100:.1f}%`")

                        else:
                            st.success(f"✅ **Diagnosis: {disease}**")
                            st.metric("Confidence", f"{confidence * 100:.1f}%")
                            st.caption(res.get("message", ""))

                            # Save history
                            st.session_state.symptom_history.append({
                                "disease": disease,
                                "confidence": f"{confidence*100:.1f}",
                                "time": datetime.now().strftime("%H:%M:%S")
                            })

                            # Confidence bar chart
                            if top3:
                                st.markdown("**Top 3 Predictions**")
                                labels = [x["disease"] for x in top3]
                                values = [round(x["confidence"] * 100, 1) for x in top3]
                                colors = ["#1D9E75", "#5DCAA5", "#9FE1CB"]

                                fig = go.Figure(go.Bar(
                                    x=values,
                                    y=labels,
                                    orientation="h",
                                    marker_color=colors,
                                    text=[f"{v}%" for v in values],
                                    textposition="outside"
                                ))
                                fig.update_layout(
                                    margin=dict(l=0, r=40, t=10, b=10),
                                    height=160,
                                    xaxis=dict(range=[0, 105], showgrid=False, visible=False),
                                    yaxis=dict(autorange="reversed"),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(size=13),
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend. Run: `uvicorn main:app --reload` from the backend folder.")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.info("Enter symptoms and click **Predict Disease** to begin.")
            st.markdown("""
**Supported Diseases (15):**
`Common Cold` · `Flu` · `COVID-19` · `Pneumonia` · `Bronchitis`
`Diabetes` · `Heart Condition` · `Hypertension` · `Anemia` · `Asthma`
`Migraine` · `Dengue` · `Typhoid` · `Gastritis` · `UTI`
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Radiology AI
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Upload Chest X-ray")
        st.caption("Upload a PA-view chest X-ray image. The CNN model will classify it as Normal, Pneumonia, or COVID-19.")

        img_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Supports JPG and PNG. Upload a grayscale or RGB chest X-ray."
        )

        if img_file:
            st.image(img_file, caption="Uploaded X-ray", use_column_width=True)
            analyze_btn = st.button("🔬 Analyze Scan", type="primary", use_container_width=True)
        else:
            analyze_btn = False
            st.info("Upload a chest X-ray image to begin analysis.")

    with col2:
        st.subheader("AI Diagnosis Report")

        if img_file and analyze_btn:
            with st.spinner("Running CNN inference..."):
                try:
                    img_file.seek(0)
                    res = requests.post(
                        f"{BACKEND}/image",
                        files={"file": (img_file.name, img_file.read(), img_file.type)},
                        timeout=30
                    ).json()

                    status = res.get("status", "")
                    prediction = res.get("prediction", "Unknown")
                    confidence = res.get("confidence", 0.0)
                    top3 = res.get("top3", [])
                    message = res.get("message", "")

                    if status == "invalid" or status == "error":
                        st.error(f"❌ **Invalid Image**")
                        st.markdown(f"> {message}")
                        st.markdown("""
**What to upload:**
- PA-view (front-facing) chest X-ray
- Clear, well-lit scan
- JPG or PNG format
- Not a regular photo or colored image
""")

                    elif status == "uncertain":
                        st.warning(f"🤔 **Uncertain — Clinical Review Recommended**")
                        st.markdown(f"> {message}")
                        st.metric("Confidence", f"{confidence * 100:.1f}%")
                        if top3:
                            st.markdown("**Model probabilities:**")
                            for item in top3:
                                st.markdown(f"- {item['label']}: `{item['confidence']*100:.1f}%`")

                    else:
                        # Color by prediction
                        color_map = {
                            "Normal": "success",
                            "Pneumonia": "warning",
                            "COVID-19": "error"
                        }
                        disp = color_map.get(prediction, "info")

                        if disp == "success":
                            st.success(f"✅ **Diagnosis: {prediction}**")
                        elif disp == "warning":
                            st.warning(f"⚠️ **Diagnosis: {prediction}**")
                        else:
                            st.error(f"🚨 **Diagnosis: {prediction}**")

                        st.metric("Model Confidence", f"{confidence * 100:.1f}%")
                        st.caption(message)

                        # Save history
                        st.session_state.image_history.append({
                            "prediction": prediction,
                            "confidence": f"{confidence*100:.1f}",
                            "time": datetime.now().strftime("%H:%M:%S")
                        })

                        # Confidence chart
                        if top3:
                            st.markdown("**CNN Class Probabilities**")
                            labels = [x["label"] for x in top3]
                            values = [round(x["confidence"] * 100, 1) for x in top3]

                            color_list = []
                            for lbl in labels:
                                if lbl == "Normal":
                                    color_list.append("#1D9E75")
                                elif lbl == "Pneumonia":
                                    color_list.append("#EF9F27")
                                else:
                                    color_list.append("#E24B4A")

                            fig = go.Figure(go.Bar(
                                x=labels,
                                y=values,
                                marker_color=color_list,
                                text=[f"{v}%" for v in values],
                                textposition="outside"
                            ))
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=10, b=10),
                                height=220,
                                yaxis=dict(range=[0, 115], showgrid=False, title="Probability (%)"),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=13),
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Run: `uvicorn main:app --reload` from the backend folder.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

        elif not img_file:
            st.markdown("""
**How it works:**
1. Upload a chest X-ray (PA view)
2. ResNet18 CNN analyzes the scan
3. Classified as Normal / Pneumonia / COVID-19
4. Confidence score + rejection logic for safety

**Safety features:**
- Blank / dark images are automatically rejected
- Colorful non-medical images are rejected
- Low-confidence results flagged for clinical review
""")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This system is for academic and research purposes only. "
    "It is not a medical device and must not be used for real clinical diagnosis. "
    "Always consult a qualified healthcare professional."
)