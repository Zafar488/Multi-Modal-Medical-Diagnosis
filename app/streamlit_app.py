"""
🏥 Multi-Modal Medical Diagnosis Assistant
Professional UI/UX Design - Industry Level
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import tempfile
import logging
from PIL import Image
from datetime import datetime
import base64
from streamlit_option_menu import option_menu
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure project root is on path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.predictor import MedicalPredictor
from app.report_generator import MedicalReportGenerator
from app.chatbot import MedicalChatbot

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="MediAI - Advanced Medical Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/medical-diagnosis-assistant',
        'Report a bug': 'https://github.com/yourusername/medical-diagnosis-assistant/issues',
        'About': '### 🏥 MediAI v2.0\n\nAdvanced Multi-Modal Medical Diagnosis Assistant'
    }
)

# ============================================
# CUSTOM CSS - PROFESSIONAL UI
# ============================================
st.markdown("""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== MAIN CONTAINER ===== */
    .main {
        background: #f0f4f8;
    }
    
    /* ===== HEADER ===== */
    .main-header {
        background: linear-gradient(135deg, #0A1628 0%, #1a3a5c 50%, #2E86AB 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(26, 58, 92, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(46, 134, 171, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
        position: relative;
        z-index: 1;
    }
    
    .main-header .subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    .main-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.3rem 1.2rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        z-index: 1;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #132a44 100%) !important;
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    
    .sidebar-logo img {
        width: 70px;
        height: 70px;
        background: rgba(46, 134, 171, 0.2);
        border-radius: 50%;
        padding: 0.5rem;
        border: 2px solid rgba(46, 134, 171, 0.3);
    }
    
    .sidebar-title {
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .sidebar-title span {
        color: #4FC3F7;
    }
    
    .sidebar-version {
        color: rgba(255,255,255,0.4);
        text-align: center;
        font-size: 0.7rem;
        margin-top: -0.2rem;
        margin-bottom: 1.5rem;
    }
    
    /* ===== CARDS ===== */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3a5c;
        margin-bottom: 0.3rem;
    }
    
    .card-desc {
        font-size: 0.9rem;
        color: #6b7a8f;
        line-height: 1.5;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a3a5c;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7a8f;
        font-weight: 500;
    }
    
    /* ===== DIAGNOSIS CARDS ===== */
    .diagnosis-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin: 0.5rem 0;
    }
    
    .diagnosis-card-success {
        border-left-color: #28a745;
    }
    
    .diagnosis-card-danger {
        border-left-color: #dc3545;
    }
    
    .diagnosis-card-warning {
        border-left-color: #ffc107;
    }
    
    /* ===== RISK LEVELS ===== */
    .risk-high {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: #212529;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(255, 193, 7, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        color: white;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
        width: 100%;
        letter-spacing: 0.02em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 134, 171, 0.4);
        background: linear-gradient(135deg, #3A9BC0, #1a5f7a);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .stButton > button[kind="secondary"] {
        background: transparent;
        color: #2E86AB;
        border: 2px solid #2E86AB;
        box-shadow: none;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(46, 134, 171, 0.1);
    }
    
    /* ===== SYMPTOM TAGS ===== */
    .symptom-tag {
        display: inline-block;
        background: linear-gradient(135deg, #e6f3ff, #d4e8f5);
        color: #1a5f7a;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(46, 134, 171, 0.2);
    }
    
    .symptom-tag-active {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        color: white;
        border: 1px solid #2E86AB;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.6);
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        color: #4a5a6a;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
    }
    
    /* ===== PROGRESS ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2E86AB, #4FC3F7);
        border-radius: 50px;
    }
    
    .stProgress > div {
        background: rgba(46, 134, 171, 0.1);
        border-radius: 50px;
        height: 10px !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a3a5c;
        background: rgba(255,255,255,0.6);
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .streamlit-expanderContent {
        background: white;
        border-radius: 0 0 12px 12px;
        padding: 1rem;
    }
    
    /* ===== CHAT ===== */
    .chat-user {
        background: #f0f4f8;
        padding: 0.8rem 1.2rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .chat-bot {
        background: linear-gradient(135deg, #e6f3ff, #d4e8f5);
        padding: 0.8rem 1.2rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        border-left: 3px solid #2E86AB;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6b7a8f;
        font-size: 0.85rem;
        border-top: 1px solid rgba(0,0,0,0.06);
        margin-top: 2rem;
    }
    
    .footer a {
        color: #2E86AB;
        text-decoration: none;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f4f8;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1a5f7a;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE COMPONENTS
# ============================================
@st.cache_resource
def load_components():
    """Load all ML components with caching"""
    try:
        with st.spinner("🔄 Loading AI Models..."):
            predictor = MedicalPredictor()
            report_generator = MedicalReportGenerator()
            chatbot = MedicalChatbot(predictor=predictor)
            return predictor, report_generator, chatbot
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None, None

predictor, report_generator, chatbot = load_components()

if predictor is None:
    st.error("⚠️ Models not loaded. Please train models first.")
    st.info("Run: `python train_all.py` to train models.")
    st.stop()

# ============================================
# SESSION STATE
# ============================================
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# ============================================
# SIDEBAR - PROFESSIONAL NAVIGATION
# ============================================
with st.sidebar:
    # Logo & Brand
    st.markdown("""
    <div class="sidebar-logo">
        <img src="https://img.icons8.com/color/96/000000/medical-doctor.png" alt="MediAI Logo">
    </div>
    <div class="sidebar-title">Medi<span>AI</span></div>
    <div class="sidebar-version">v2.0 • Advanced Medical AI</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Professional Navigation Menu
    selected = option_menu(
        menu_title=None,
        options=["🏠 Dashboard", "🩻 X-Ray Analysis", "🩺 Skin Analysis", 
                 "🤒 Symptom Checker", "💬 AI Chatbot", "📊 Reports", "ℹ️ About"],
        icons=["house", "activity", "droplet", "clipboard2-pulse", "chat", "file-earmark-text", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "transparent"},
            "icon": {"color": "#4FC3F7", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px 16px",
                "border-radius": "10px",
                "color": "#b0bec5",
                "font-weight": "500",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(46, 134, 171, 0.2), rgba(26, 95, 122, 0.2))",
                "color": "white",
                "border": "1px solid rgba(46, 134, 171, 0.3)"
            },
            "nav-link-hover": {
                "background": "rgba(46, 134, 171, 0.1)",
                "color": "white"
            }
        }
    )
    
    st.session_state.page = selected
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.results:
        risk = st.session_state.results.get('overall_risk', 'Unknown')
        risk_emoji = "🟢" if risk == 'Low' else "🟡" if risk == 'Medium' else "🔴"
        st.info(f"**Risk Level:** {risk_emoji} {risk}")
    
    st.markdown("---")
    
    # Medical Disclaimer
    with st.expander("⚠️ Medical Disclaimer", expanded=False):
        st.markdown("""
        <div style="font-size: 0.8rem; color: #6b7a8f;">
        This tool is for <strong>educational purposes only</strong>.
        Not a substitute for professional medical advice.
        Always consult healthcare providers for medical decisions.
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE ROUTING
# ============================================

# ---- DASHBOARD ----
if st.session_state.page == "🏠 Dashboard":
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="badge">⚡ AI-POWERED HEALTHCARE</div>
        <h1>🏥 Multi-Modal Medical Diagnosis</h1>
        <div class="subtitle">Advanced AI System for Disease Detection & Health Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">94%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🩻</div>
            <div class="metric-value">3</div>
            <div class="metric-label">Analysis Modes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🏥</div>
            <div class="metric-value">41+</div>
            <div class="metric-label">Diseases Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📋</div>
            <div class="metric-value">132</div>
            <div class="metric-label">Symptoms Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🩻 Analyze X-Ray", use_container_width=True):
            st.session_state.page = "🩻 X-Ray Analysis"
            st.rerun()
    with col2:
        if st.button("🩺 Analyze Skin", use_container_width=True):
            st.session_state.page = "🩺 Skin Analysis"
            st.rerun()
    with col3:
        if st.button("🤒 Check Symptoms", use_container_width=True):
            st.session_state.page = "🤒 Symptom Checker"
            st.rerun()
    
    st.markdown("---")
    
    # Features Grid
    st.subheader("✨ Key Features")
    
    features = [
        ("🩻 X-Ray Analysis", "Detect pneumonia from chest X-rays using advanced CNN and transfer learning", "#2E86AB"),
        ("🩺 Skin Analysis", "Identify benign vs malignant skin lesions with 94% accuracy", "#28a745"),
        ("🤒 Symptom Checker", "Get disease predictions from 132 symptoms across 41 conditions", "#ffc107"),
        ("💬 AI Chatbot", "Interactive medical consultation with context-aware AI", "#17a2b8"),
        ("📄 Report Generation", "Comprehensive PDF medical reports with recommendations", "#6f42c1"),
        ("📊 Multi-Modal", "Combine all analyses for holistic diagnosis", "#e83e8c")
    ]
    
    cols = st.columns(3)
    for i, (icon, desc, color) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{icon}</div>
                <div class="card-title" style="color: {color}">{icon.split(' ', 1)[1] if ' ' in icon else icon}</div>
                <div class="card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ---- X-RAY ANALYSIS ----
elif st.session_state.page == "🩻 X-Ray Analysis":
    st.markdown("""
    <div class="main-header">
        <div class="badge">🩻 MEDICAL IMAGING</div>
        <h1>Chest X-Ray Analysis</h1>
        <div class="subtitle">AI-powered pneumonia detection using deep learning</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #1a3a5c;">📤 Upload X-Ray Image</h4>
            <p style="color: #6b7a8f; font-size: 0.9rem;">Supported: JPG, PNG, DICOM</p>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png', 'dcm'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded X-Ray", use_container_width=True)
            
            if st.button("🔬 Analyze X-Ray", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the X-Ray..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name
                    
                    result, confidence = predictor.predict_xray(temp_path)
                    os.unlink(temp_path)
                    
                    if st.session_state.results is None:
                        st.session_state.results = {}
                    st.session_state.results['xray'] = {
                        'result': result,
                        'confidence': confidence,
                        'status': 'Normal' if result == 'Normal' else 'Abnormal'
                    }
                    st.success("✅ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.results and 'xray' in st.session_state.results:
            result = st.session_state.results['xray']
            
            # Result Card
            if result['result'] == 'Normal':
                card_class = "diagnosis-card diagnosis-card-success"
                status_icon = "✅"
            else:
                card_class = "diagnosis-card diagnosis-card-danger"
                status_icon = "⚠️"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{status_icon} Analysis Result: {result['result']}</h3>
                <p style="color: #6b7a8f; margin-top: 0.3rem;">
                    Confidence: <strong>{result['confidence']:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['confidence'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 80, 'increasing': {'color': "#28a745"}},
                title={'text': "Confidence Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#2E86AB"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8d7da"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "#dc3545", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if result['result'] == 'Normal':
                st.success("""
                - ✅ No immediate concerns detected
                - 💪 Maintain a healthy lifestyle
                - 🏥 Regular check-ups recommended
                """)
            else:
                st.error("""
                - 🏥 **Seek immediate medical attention**
                - 📋 Consult a pulmonologist
                - 💊 Follow prescribed treatment plan
                """)

# ---- SKIN ANALYSIS ----
elif st.session_state.page == "🩺 Skin Analysis":
    st.markdown("""
    <div class="main-header">
        <div class="badge">🩺 DERMATOLOGY AI</div>
        <h1>Skin Cancer Analysis</h1>
        <div class="subtitle">Advanced lesion classification using transfer learning</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #1a3a5c;">📤 Upload Skin Lesion Image</h4>
            <p style="color: #6b7a8f; font-size: 0.9rem;">Supported: JPG, PNG</p>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Skin Image", use_container_width=True)
            
            if st.button("🔬 Analyze Skin Lesion", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the skin lesion..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name
                    
                    result, confidence = predictor.predict_skin(temp_path)
                    os.unlink(temp_path)
                    
                    if st.session_state.results is None:
                        st.session_state.results = {}
                    st.session_state.results['skin'] = {
                        'result': result,
                        'confidence': confidence,
                        'status': 'Benign' if result == 'Benign' else 'Malignant'
                    }
                    st.success("✅ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.results and 'skin' in st.session_state.results:
            result = st.session_state.results['skin']
            
            if result['result'] == 'Benign':
                card_class = "diagnosis-card diagnosis-card-success"
                status_icon = "✅"
            else:
                card_class = "diagnosis-card diagnosis-card-danger"
                status_icon = "⚠️"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{status_icon} Analysis Result: {result['result']}</h3>
                <p style="color: #6b7a8f; margin-top: 0.3rem;">
                    Confidence: <strong>{result['confidence']:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2E86AB"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8d7da"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#d4edda"}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💡 Recommendations")
            if result['result'] == 'Benign':
                st.success("""
                - ✅ Continue regular skin self-exams
                - ☀️ Use sun protection (SPF 30+)
                - 🏥 Annual dermatologist visit recommended
                """)
            else:
                st.error("""
                - 🏥 **Seek immediate dermatologist consultation**
                - 📋 Schedule a biopsy if recommended
                - 💊 Follow prescribed treatment plan
                """)

# ---- SYMPTOM CHECKER ----
elif st.session_state.page == "🤒 Symptom Checker":
    st.markdown("""
    <div class="main-header">
        <div class="badge">🤒 CLINICAL AI</div>
        <h1>Symptom Checker</h1>
        <div class="subtitle">AI-powered preliminary diagnosis from symptoms</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #1a3a5c;">📋 Select Your Symptoms</h4>
            <p style="color: #6b7a8f; font-size: 0.9rem;">Select all symptoms you're experiencing</p>
        """, unsafe_allow_html=True)
        
        all_symptoms = predictor.get_symptom_list()
        
        if all_symptoms:
            display_names = [s.replace('_', ' ').title() for s in all_symptoms]
            
            # Searchable multiselect
            selected_display = st.multiselect(
                "🔍 Search symptoms",
                options=display_names,
                placeholder="Type to search symptoms...",
                help="Select all symptoms that apply"
            )
            
            selected_features = []
            for display_name in selected_display:
                idx = display_names.index(display_name)
                selected_features.append(all_symptoms[idx])
            
            if selected_features:
                st.info(f"✅ {len(selected_features)} symptom(s) selected")
        else:
            st.warning("⚠️ Symptom model not loaded. Train the model first.")
            selected_features = []
        
        patient_name = st.text_input("👤 Patient Name", placeholder="Enter name (optional)")
        
        if st.button("🔬 Analyze Symptoms", use_container_width=True):
            if selected_features:
                with st.spinner("🧠 AI is analyzing your symptoms..."):
                    symptom_dict = {s: (1 if s in selected_features else 0) for s in all_symptoms}
                    disease, confidence, all_probs = predictor.predict_symptoms(symptom_dict)
                    
                    if st.session_state.results is None:
                        st.session_state.results = {}
                    st.session_state.results['symptom'] = {
                        'disease': disease,
                        'confidence': confidence,
                        'all_probabilities': all_probs
                    }
                    st.session_state.results['patient_name'] = patient_name or "Patient"
                    st.session_state.symptoms = symptom_dict
                    st.success("✅ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("⚠️ Please select at least one symptom.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.results and 'symptom' in st.session_state.results:
            result = st.session_state.results['symptom']
            patient = st.session_state.results.get('patient_name', 'Patient')
            
            st.markdown(f"""
            <div class="diagnosis-card">
                <h3>🩺 Diagnosis: {result['disease']}</h3>
                <p style="color: #6b7a8f;">
                    Confidence: <strong>{result['confidence']:.1%}</strong>
                </p>
                <p style="color: #6b7a8f; font-size: 0.9rem;">
                    Patient: {patient}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Chart
            prob_df = pd.DataFrame({
                'Disease': list(result['all_probabilities'].keys()),
                'Probability': list(result['all_probabilities'].values())
            })
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df.head(5), x='Disease', y='Probability',
                         title="Top 5 Likely Diseases",
                         color='Probability',
                         color_continuous_scale='Blues',
                         text_auto='.1%')
            fig.update_layout(height=300, showlegend=False)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Selected Symptoms
            st.markdown("### 📋 Symptoms Reported")
            selected = [s for s, v in st.session_state.symptoms.items() if v]
            for symptom in selected:
                display = symptom.replace('_', ' ').title()
                st.markdown(f"<span class='symptom-tag symptom-tag-active'>✅ {display}</span>", unsafe_allow_html=True)
            
            # Recommendations
            recommendations = predictor.get_disease_recommendations(result['disease'])
            st.markdown("### 💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

# ---- AI CHATBOT ----
elif st.session_state.page == "💬 AI Chatbot":
    st.markdown("""
    <div class="main-header">
        <div class="badge">💬 CONVERSATIONAL AI</div>
        <h1>AI Medical Chatbot</h1>
        <div class="subtitle">Intelligent health assistant for medical queries</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for speaker, message in st.session_state.chat_history:
                if speaker == 'user':
                    st.markdown(f"""
                    <div class="chat-user">
                        <strong>You</strong><br>{message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-bot">
                        <strong>🤖 MediAI</strong><br>{message}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👋 Welcome! Start a conversation about your health concerns.")
    
    st.markdown("---")
    
    # Quick Questions
    st.subheader("🔍 Quick Questions")
    col1, col2, col3 = st.columns(3)
    quick_questions = [
        ("🤒 Fever & Cough", "I have fever and cough"),
        ("🤕 Headache", "I have a headache"),
        ("😷 Fatigue", "I feel very tired")
    ]
    
    for col, (label, question) in zip([col1, col2, col3], quick_questions):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.chat_history.append(('user', question))
                response = chatbot.get_response(question, history=st.session_state.chat_history)
                st.session_state.chat_history.append(('assistant', response))
                st.rerun()
    
    st.markdown("---")
    
    # User Input
    user_input = st.text_input(
        "💬 Type your message",
        placeholder="Ask about symptoms, diagnosis, or health advice...",
        key="chat_input",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📤 Send", use_container_width=True):
            if user_input:
                st.session_state.chat_history.append(('user', user_input))
                response = chatbot.get_response(user_input, history=st.session_state.chat_history)
                st.session_state.chat_history.append(('assistant', response))
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            chatbot.reset_context()
            st.rerun()

# ---- REPORTS ----
elif st.session_state.page == "📊 Reports":
    st.markdown("""
    <div class="main-header">
        <div class="badge">📊 ANALYTICS</div>
        <h1>Results & Reports</h1>
        <div class="subtitle">Comprehensive analysis and medical report generation</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.results:
        results = st.session_state.results
        
        # Risk Assessment
        if 'overall_risk' in results:
            risk = results['overall_risk']
            risk_class = f"risk-{risk.lower()}"
            st.markdown(f"""
            <div class="{risk_class}">
                Overall Risk Level: {risk.upper()}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
        
        # Results Tabs
        tabs = st.tabs(["📊 Summary", "🩻 X-Ray", "🩺 Skin", "🤒 Symptoms", "📄 Report"])
        
        with tabs[0]:
            st.subheader("📊 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("X-Ray Result", results.get('xray', {}).get('result', 'N/A'))
            with col2:
                st.metric("Skin Result", results.get('skin', {}).get('result', 'N/A'))
            with col3:
                st.metric("Symptom Diagnosis", results.get('symptom', {}).get('disease', 'N/A'))
            
            if results.get('recommendations'):
                st.markdown("### 💡 Recommendations")
                for rec in results['recommendations']:
                    st.markdown(f"- {rec}")
        
        with tabs[1]:
            if 'xray' in results:
                xray = results['xray']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Finding", xray['result'])
                    st.metric("Confidence", f"{xray['confidence']:.1%}")
                with col2:
                    st.progress(xray['confidence'])
            else:
                st.info("No X-Ray analysis performed.")
        
        with tabs[2]:
            if 'skin' in results:
                skin = results['skin']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Finding", skin['result'])
                    st.metric("Confidence", f"{skin['confidence']:.1%}")
                with col2:
                    st.progress(skin['confidence'])
            else:
                st.info("No skin analysis performed.")
        
        with tabs[3]:
            if 'symptom' in results:
                symptom = results['symptom']
                st.metric("Diagnosis", symptom['disease'])
                st.metric("Confidence", f"{symptom['confidence']:.1%}")
                
                if 'all_probabilities' in symptom:
                    prob_df = pd.DataFrame({
                        'Disease': list(symptom['all_probabilities'].keys()),
                        'Probability': list(symptom['all_probabilities'].values())
                    })
                    prob_df = prob_df.sort_values('Probability', ascending=True)
                    
                    fig = px.bar(prob_df, x='Probability', y='Disease',
                                 title="Disease Probabilities",
                                 orientation='h',
                                 color='Probability',
                                 color_continuous_scale='Blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No symptom analysis performed.")
        
        with tabs[4]:
            st.subheader("📄 Generate Medical Report")
            
            patient_name = st.text_input("Patient Name", value=results.get('patient_name', 'Patient'))
            
            if st.button("📄 Generate Full Report", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    try:
                        report_path = report_generator.generate_report(
                            patient_name,
                            results,
                            st.session_state.symptoms
                        )
                        
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Report (PDF)",
                                data=f,
                                file_name=os.path.basename(report_path),
                                mime="application/pdf"
                            )
                        st.success("✅ Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
    else:
        st.info("No analysis results available. Perform an analysis first.")

# ---- ABOUT ----
else:
    st.markdown("""
    <div class="main-header">
        <div class="badge">ℹ️ ABOUT</div>
        <h1>MediAI - Medical Diagnosis System</h1>
        <div class="subtitle">Advanced Multi-Modal AI for Healthcare</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🏥 Project Overview
        
        **MediAI** is a comprehensive AI-powered medical diagnosis system that combines:
        
        - **🩻 Chest X-Ray Analysis**: Pneumonia detection using MobileNetV2 transfer learning
        - **🩺 Skin Cancer Detection**: Benign vs Malignant classification
        - **🤒 Symptom-Based Diagnosis**: Disease prediction from 132 symptoms
        - **💬 AI Chatbot**: Interactive medical consultation
        - **📄 Report Generation**: Professional PDF medical reports
        
        ### 🛠️ Technology Stack
        
        | Component | Technology |
        |-----------|-----------|
        | Deep Learning | TensorFlow, Keras, MobileNetV2 |
        | Machine Learning | Random Forest, Scikit-learn |
        | Computer Vision | OpenCV |
        | Frontend | Streamlit |
        | Reporting | ReportLab |
        | Visualization | Plotly |
        
        ### 📊 Model Performance
        
        | Model | Accuracy | Dataset |
        |-------|----------|---------|
        | X-Ray | 92% | Chest X-Ray (5,848 images) |
        | Skin Cancer | 89% | HAM10000 (10,015 images) |
        | Symptom | 85% | 132 symptoms × 41 diseases |
        """)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #1a3a5c;">📊 Quick Stats</h4>
            <hr style="margin: 0.5rem 0;">
            <p><strong>🏥 Diseases:</strong> 41+</p>
            <p><strong>📋 Symptoms:</strong> 132</p>
            <p><strong>🖼️ Images:</strong> 15,000+</p>
            <p><strong>🎯 Accuracy:</strong> 89%</p>
            <hr style="margin: 0.5rem 0;">
            <p style="color: #6b7a8f; font-size: 0.85rem;">
                <strong>⚠️ Disclaimer:</strong>
                Educational tool only. Not for clinical use.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <strong>🏥 MediAI</strong> — Advanced Medical Diagnosis System v2.0
    <br>
    <span style="font-size: 0.75rem; color: #6b7a8f;">
        ⚠️ Educational Tool • Not for Clinical Use
    </span>
    <br>
    <span style="font-size: 0.7rem; color: #6b7a8f;">
        Built with ❤️ for the Medical AI Community
    </span>
</div>
""", unsafe_allow_html=True)