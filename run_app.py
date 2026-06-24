#!/usr/bin/env python3
"""
Multi-Modal Medical Diagnosis Assistant
Main entry point for the application
"""

import sys
import os
import subprocess


def check_virtual_env():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("⚠️ Warning: Not running in a virtual environment!")
        print("💡 It's recommended to use a virtual environment.")
    else:
        print("✅ Running in virtual environment")
        print(f"📁 Python Path: {sys.prefix}")


def check_requirements():
    """Check if all requirements are installed"""
    try:
        import tensorflow as tf
        import streamlit
        import pandas
        import numpy
        import cv2
        print("✅ All major dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install requirements: pip install -r requirements.txt")
        return False


def check_models():
    """Check if trained models exist"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    models = [
        'models/disease_prediction_model.pkl',
        'models/xray_model.h5',
        'models/skin_model.h5'
    ]

    missing = []
    for model in models:
        path = os.path.join(project_root, model)
        if os.path.exists(path):
            print(f"✅ Found: {model}")
        else:
            print(f"⚠️ Missing: {model}")
            missing.append(model)

    if missing:
        print(f"\n⚠️ {len(missing)} model(s) missing. Run training first:")
        print("   python -m training.train_symptom")
        print("   python -m training.train_xray")
        print("   python -m training.train_skin")

    return len(missing) == 0


def main():
    """Launch the Streamlit application"""
    print("=" * 70)
    print("🏥 Multi-Modal Medical Diagnosis Assistant v1.0")
    print("=" * 70)

    check_virtual_env()

    if not check_requirements():
        sys.exit(1)

    check_models()

    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
    app_file = os.path.join(app_dir, 'streamlit_app.py')

    if not os.path.exists(app_file):
        print(f"❌ Error: {app_file} not found!")
        sys.exit(1)

    print(f"\n📁 Loading application from: {app_file}")
    print("🌐 Application will be available at: http://localhost:8501")
    print("=" * 70)
    print("\n🚀 Starting Streamlit Application...\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Try running: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()