# src/streamlit_app.py
"""
Enhanced Streamlit UI with permanent icon fixes using native emojis
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from streamlit_lottie import st_lottie
import requests
import io
import tempfile
import os
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# Configuration & Initialization
# -----------------------------
st.set_page_config(
    page_title="Diabetes — Modern UI", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    defaults = {
        'preg': 0, 'gluc': 120, 'bp': 70, 'skin': 20,
        'ins': 80, 'bmi': 25.0, 'dpf': 0.5, 'age': 30,
        'initialized': True,
        'form_submitted': False,
        'clear_triggered': False,
        'demo_triggered': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ------------- Helper Functions -------------
@st.cache_data(show_spinner=False)
def load_lottie_url(url: str, timeout: int = 5):
    """Cache Lottie animations to prevent re-downloads"""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """Cache model and scaler loading"""
    MODEL_PATHS = [
        Path("model/diabetes_model.joblib"), Path("model/diabetes_model.pkl"),
        Path("models/diabetes_model.joblib"), Path("models/diabetes_model.pkl"),
    ]
    SCALER_PATHS = [
        Path("model/scaler.joblib"), Path("model/scaler.pkl"),
        Path("models/scaler.joblib"), Path("models/scaler.pkl"),
    ]
    
    model_path = next((p for p in MODEL_PATHS if p.exists()), None)
    scaler_path = next((p for p in SCALER_PATHS if p.exists()), None)

    if model_path is None or scaler_path is None:
        st.error("Model or scaler not found. Place your trained model & scaler in `model/` or `models/`.")
        st.stop()
    
    return joblib.load(model_path), joblib.load(scaler_path)

def safe_float_fmt(v, ndigits=3):
    """Safely format float values"""
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return "N/A"

def calc_metrics_from_probs(y_true, y_prob, threshold=0.5):
    """Calculate model metrics"""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    except Exception:
        metrics = {k: None for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        metrics["confusion_matrix"] = [[0, 0], [0, 0]]
    return metrics

def _safe_write_image(fig, path):
    """Safely write plotly figures to images"""
    try:
        fig.write_image(path, format="png", width=900, height=520, scale=2)
    except Exception:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Image export failed", ha="center", va="center")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

def generate_pdf_report(patient_info, pred_text, metrics, roc_fig, cm_fig):
    """Generate PDF report with error handling"""
    tmpdir = tempfile.mkdtemp()
    roc_path = os.path.join(tmpdir, "roc.png")
    cm_path = os.path.join(tmpdir, "cm.png")
    
    try:
        _safe_write_image(roc_fig, roc_path)
        _safe_write_image(cm_fig, cm_path)

        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Diabetes Prediction Report", ln=True, align="C")
        pdf.ln(6)

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Prediction: {pred_text}", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Patient Inputs:", ln=True)
        pdf.set_font("Arial", size=11)
        for k, v in patient_info.items():
            pdf.cell(0, 7, f"{k}: {v}", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Model Metrics:", ln=True)
        pdf.set_font("Arial", size=11)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            val = metrics.get(key)
            if val is None:
                pdf.cell(0, 7, f"{key.capitalize()}: N/A", ln=True)
            else:
                pdf.cell(0, 7, f"{key.capitalize()}: {val:.4f}", ln=True)
        pdf.ln(6)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "ROC Curve", ln=True)
        pdf.image(roc_path, w=180)
        pdf.ln(6)
        pdf.cell(0, 7, "Confusion Matrix", ln=True)
        pdf.image(cm_path, w=140)

        pdf_output = pdf.output(dest='S').encode('latin-1')
        bio = io.BytesIO(pdf_output)
        return bio.getvalue()
        
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return b""
    finally:
        try:
            if os.path.exists(roc_path):
                os.remove(roc_path)
            if os.path.exists(cm_path):
                os.remove(cm_path)
            os.rmdir(tmpdir)
        except Exception:
            pass

# -----------------------------
# Enhanced CSS (Clean - No External Dependencies)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #F7F7F7 0%, #FFFFFF 50%, #FFF5F2 100%);
    color: #1a1a1a;
}

/* Emoji spacing */
.emoji-icon {
    margin-right: 8px;
    display: inline-block;
}

/* Card Styles */
.card {
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(255, 255, 255, 0.4);
    transition: all 0.3s ease;
    margin-bottom: 16px;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}

/* Button Styles */
.stButton > button {
    background: linear-gradient(135deg, #FF5733 0%, #FF8C42 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    border: none !important;
    font-weight: 600;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255,87,51,0.3);
}

/* KPI Cards */
.kpi-card {
    text-align: center;
    padding: 16px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.1);
}

.kpi-card h3 {
    margin: 0;
    font-size: 12px;
    color: #666;
    font-weight: 600;
    text-transform: uppercase;
}

.kpi-card h2 {
    margin: 8px 0 0 0;
    font-size: 20px;
    color: #1a1a1a;
    font-weight: 700;
}

/* Prediction Results */
.prediction-positive {
    background: rgba(255, 80, 80, 0.28);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    margin: 16px 0;
    color: #000;
    border: 1px solid rgba(255, 150, 150, 0.45);
    box-shadow: 0 6px 18px rgba(255, 90, 90, 0.4);
}

.prediction-negative {
    background: rgba(0, 220, 180, 0.28);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    margin: 16px 0;
    color: #000;
    border: 1px solid rgba(120, 255, 230, 0.4);
    box-shadow: 0 6px 18px rgba(0, 220, 180, 0.4);
}

/* Progress Bars */
.progress-container {
    margin: 12px 0;
}

.progress-bar {
    height: 6px;
    border-radius: 3px;
    background: #e0e0e0;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF5733, #FF8C42);
    border-radius: 3px;
    transition: width 0.8s ease-in-out;
}

/* Metric Indicators */
.metric-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}

.metric-high { background: #4ECDC4; }
.metric-medium { background: #FFD166; }
.metric-low { background: #FF6B6B; }

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in { animation: fadeIn 0.5s ease both; }

/* Download Button */
.download-btn {
    background: linear-gradient(135deg, #4ECDC4, #6BFFE6) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    border: none !important;
    font-weight: 600;
    margin: 20px 0;
}

/* Status styling */
.status-positive { color: #4ECDC4; }
.status-warning { color: #FF6B6B; }
.status-info { color: #FFD166; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model & Data
# -----------------------------
model, scaler = load_model_and_scaler()
metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
test_exists = False

# -----------------------------
# Sidebar Configuration WITH EMOJI ICONS
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    threshold = st.slider(
        "Decision Threshold", 
        0.0, 1.0, 0.5, 0.01,
        help="Adjust the probability threshold for classification",
        key="threshold_slider"
    )
    
    st.markdown("---")
    st.markdown("### 📝 Test Data")
    
    uploaded_X = st.file_uploader("Upload X_test.csv", type=["csv"], key="x_upload")
    uploaded_y = st.file_uploader("Upload y_test.csv", type=["csv"], key="y_upload")
    
    if uploaded_X and uploaded_y:
        try:
            X_test = pd.read_csv(uploaded_X)
            y_test = pd.read_csv(uploaded_y).squeeze()
            X_scaled = scaler.transform(X_test)
            y_test_proba = model.predict_proba(X_scaled)[:, 1]
            metrics = calc_metrics_from_probs(y_test, y_test_proba, threshold)
            test_exists = True
        except Exception as e:
            st.error(f"Error processing test data: {e}")

# -----------------------------
# Header Section WITH EMOJI ICONS
# -----------------------------
col1, col2 = st.columns([1, 4])
with col1:
    lottie_json = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json")
    if lottie_json:
        st_lottie(lottie_json, height=100, key="header_animation")

with col2:
    st.markdown("""
        <div style='padding: 10px 0;'>
            <h1 style='margin: 0; color: #1a1a1a; font-weight: 800;'>
                🩺 Diabetes Prediction System
            </h1>
            <p style='margin: 5px 0 0 0; color: #666; font-size: 1.1rem;'>
                     Advanced AI •  Clinical Analytics •  Real-time Assessment
            </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# KPI Dashboard WITH EMOJI ICONS
# -----------------------------
st.markdown("###  Performance Metrics")
cols = st.columns(4)

metric_configs = [
    ("Accuracy", "accuracy", "#4ECDC4", "🎯"),
    ("ROC AUC", "roc_auc", "#FFD166", "📊"), 
    ("Precision", "precision", "#FF9E6D", "🎯"),
    ("Recall", "recall", "#6BFFE6", "🔍")
]

for idx, (title, key, color, icon) in enumerate(metric_configs):
    with cols[idx]:
        value = metrics.get(key, 0)
        metric_class = "metric-high" if value and value > 0.8 else "metric-medium" if value and value > 0.6 else "metric-low"
        
        st.markdown(f"""
            <div class='kpi-card fade-in'>
                <span class='metric-indicator {metric_class}'></span>
                <h3>{icon} {title}</h3>
                <h2>{safe_float_fmt(value)}</h2>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Handle Clear and Demo Actions
# -----------------------------
if st.session_state.get('clear_triggered', False):
    st.session_state.preg = 0
    st.session_state.gluc = 120
    st.session_state.bp = 70
    st.session_state.skin = 20
    st.session_state.ins = 80
    st.session_state.bmi = 25.0
    st.session_state.dpf = 0.5
    st.session_state.age = 30
    st.session_state.clear_triggered = False
    st.session_state.form_submitted = False
    st.rerun()

if st.session_state.get('demo_triggered', False):
    st.session_state.preg = 2
    st.session_state.gluc = 140
    st.session_state.bp = 80
    st.session_state.skin = 25
    st.session_state.ins = 100
    st.session_state.bmi = 28.5
    st.session_state.dpf = 0.5
    st.session_state.age = 45
    st.session_state.demo_triggered = False
    st.session_state.form_submitted = False
    st.rerun()

# -----------------------------
# Patient Input Form WITH EMOJI ICONS
# -----------------------------
st.markdown("###  Patient Assessment")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    with st.form("patient_form", clear_on_submit=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**👥 Demographics**")
            pregnancies = st.number_input("Pregnancies", 0, 20, key='preg', help="Number of pregnancies")
            age = st.slider("Age", 1, 120, key='age', help="Patient age")
            
        with col2:
            st.markdown("**🩸 Blood Work**")
            glucose = st.slider("Glucose", 0, 300, key='gluc', help="Glucose level in mg/dL")
            insulin = st.slider("Insulin", 0, 900, key='ins', help="Insulin level in μU/mL")
            
        with col3:
            st.markdown("**❤️ Physical Metrics**")
            bp = st.slider("Blood Pressure", 0, 200, key='bp', help="Diastolic blood pressure")
            skin = st.slider("Skin Thickness", 0, 100, key='skin', help="Triceps skin fold thickness")
            
        with col4:
            st.markdown("**⚖️ Body Composition**")
            bmi = st.slider("BMI", 0.0, 70.0, key='bmi', help="Body Mass Index")
            dpf = st.slider("DPF", 0.0, 3.0, key='dpf', help="Diabetes Pedigree Function")
        
        # Form actions with emojis
        submit_col1, submit_col2, submit_col3 = st.columns([2, 1, 1])
        with submit_col1:
            submit_prediction = st.form_submit_button(
                "🚨 Predict Diabetes Risk", 
                use_container_width=True
            )
        
        with submit_col2:
            clear_clicked = st.form_submit_button(
                "🧹 Clear", 
                use_container_width=True
            )
            if clear_clicked:
                st.session_state.clear_triggered = True
                st.rerun()
        
        with submit_col3:
            demo_clicked = st.form_submit_button(
                "💡 Demo", 
                use_container_width=True
            )
            if demo_clicked:
                st.session_state.demo_triggered = True
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Results WITH EMOJI ICONS
# -----------------------------
if submit_prediction:
    st.session_state.form_submitted = True
    
    features = np.array([[
        st.session_state['preg'], st.session_state['gluc'], st.session_state['bp'],
        st.session_state['skin'], st.session_state['ins'], st.session_state['bmi'],
        st.session_state['dpf'], st.session_state['age']
    ]])
    
    try:
        scaled_features = scaler.transform(features)
        probability = float(model.predict_proba(scaled_features)[0, 1])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        probability = 0.0
    
    prediction = "Diabetic" if probability >= threshold else "Not Diabetic"
    risk_level = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.3 else "LOW"
    
    # Display prediction result with emoji icons
    if prediction == "Diabetic":
        st.markdown(f"""
            <div class='prediction-positive fade-in'>
                <h2 style='margin: 0;'>
                    ⚠️ {prediction}
                </h2>
                <p style='margin: 10px 0; font-size: 1.1rem;'>
                    🚩 Risk Level: <strong>{risk_level}</strong> • 
                    📊 Probability: <strong>{probability:.3f}</strong>
                </p>
                <p style='margin: 0; font-size: 0.9rem;'>
                    ⚙️ Threshold: {threshold:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='prediction-negative fade-in'>
                <h2 style='margin: 0;'>
                    ✅ {prediction}
                </h2>
                <p style='margin: 10px 0; font-size: 1.1rem;'>
                    🚩 Risk Level: <strong>{risk_level}</strong> • 
                    📊 Probability: <strong>{probability:.3f}</strong>
                </p>
                <p style='margin: 0; font-size: 0.9rem;'>
                    ⚙️ Threshold: {threshold:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualization Tabs with emoji icons
    tab1, tab2, tab3 = st.tabs([
        "📊 Feature Analysis", 
        "🔬 Model Insights", 
        "👤 Patient Profile"
    ])
    
    with tab1:
        if hasattr(model, 'feature_importances_'):
            try:
                feature_names = ['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age']
                importances = model.feature_importances_
                
                fig = px.bar(
                    x=importances, y=feature_names, 
                    orientation='h',
                    title="📈 Feature Importance Analysis",
                    labels={'x': 'Importance', 'y': 'Features'}
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception:
                st.info("ℹ️ Feature importance not available for this model.")
        else:
            st.info("ℹ️ This model doesn't support feature importance visualization.")
    
    with tab2:
        if test_exists:
            col1, col2 = st.columns(2)
            with col1:
                fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                roc_fig = px.area(
                    x=fpr, y=tpr,
                    title=f'📈 ROC Curve (AUC = {metrics["roc_auc"]:.3f})',
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                )
                roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(roc_fig, use_container_width=True)
            
            with col2:
                cm = metrics['confusion_matrix']
                cm_fig = px.imshow(
                    cm,
                    text_auto=True,
                    title='📊 Confusion Matrix',
                    labels={'x': 'Predicted', 'y': 'Actual'}
                )
                st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.info('ℹ️ Upload test data to see model performance visualizations.')
    
    with tab3:
        patient_data = {
            'Metric': ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Insulin', 'Skin Thickness'],
            'Value': [
                st.session_state['gluc'], st.session_state['bmi'], st.session_state['age'],
                st.session_state['bp'], st.session_state['ins'], st.session_state['skin']
            ]
        }
        
        summary_fig = px.bar(
            patient_data, x='Value', y='Metric', orientation='h',
            title='📋 Patient Health Metrics',
            color='Value', color_continuous_scale='OrRd'
        )
        st.plotly_chart(summary_fig, use_container_width=True)

    # PDF Report Generation
    st.markdown("---")
    st.markdown("### 📄 Download Report")
    
    patient_info = {
        'Pregnancies': st.session_state['preg'],
        'Glucose': st.session_state['gluc'],
        'Blood Pressure': st.session_state['bp'],
        'Skin Thickness': st.session_state['skin'],
        'Insulin': st.session_state['ins'],
        'BMI': st.session_state['bmi'],
        'Diabetes Pedigree Function': st.session_state['dpf'],
        'Age': st.session_state['age']
    }
    
    # Create visualizations for PDF
    if test_exists:
        roc_fig_pdf = go.Figure()
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_fig_pdf.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#FF5733')))
        roc_fig_pdf.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
        
        cm_fig_pdf = go.Figure(data=go.Heatmap(
            z=metrics['confusion_matrix'],
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues'
        ))
    else:
        roc_fig_pdf = go.Figure()
        roc_fig_pdf.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='ROC Curve', line=dict(color='#FF5733')))
        
        cm_fig_pdf = go.Figure(data=go.Heatmap(
            z=[[0,0],[0,0]],
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues'
        ))
    
    pdf_bytes = generate_pdf_report(patient_info, f"{prediction} (Risk: {risk_level})", metrics, roc_fig_pdf, cm_fig_pdf)
    
    if pdf_bytes:
        st.download_button(
            label=" Download Comprehensive PDF Report",
            data=pdf_bytes,
            file_name=f"diabetes_prediction_report_{st.session_state['age']}y_{probability:.2f}risk.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download"
        )
    else:
        st.error(" Failed to generate PDF report. Please try again.")

# -----------------------------
# Footer WITH EMOJI ICONS
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "<p>❤️ <strong>Diabetes Prediction System</strong> • Built for Clinical Decision Support</p>"
    "<p style='font-size: 0.9rem;'>🎓 For educational and research purposes. Consult healthcare professionals for medical decisions.</p>"
    "</div>",
    unsafe_allow_html=True
)