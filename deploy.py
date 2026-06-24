"""
Deployment script for Multi-Modal Medical Diagnosis Assistant
Handles directory creation, model training, and app launch
"""
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def create_directories():
    """Create all necessary directories"""
    print("\n📁 Creating directory structure...")
    directories = [
        'models',
        'reports/generated',
    ]

    for directory in directories:
        full_path = os.path.join(PROJECT_ROOT, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"📁 Created: {directory}")


def train_models():
    """Train all models"""
    print("\n🤖 Training models...")

    # Train symptom model first (fastest)
    print("\n--- Training Symptom Model ---")
    try:
        from training.train_symptom import SymptomTrainer
        trainer = SymptomTrainer()
        trainer.train()
    except Exception as e:
        print(f"⚠️ Symptom training failed: {e}")

    # Train X-Ray model
    print("\n--- Training X-Ray Model ---")
    try:
        from training.train_xray import ChestXRayTrainer
        trainer = ChestXRayTrainer()
        trainer.train(epochs=5)
    except Exception as e:
        print(f"⚠️ X-Ray training failed: {e}")

    # Train Skin Cancer model
    print("\n--- Training Skin Cancer Model ---")
    try:
        from training.train_skin import SkinCancerTrainer
        trainer = SkinCancerTrainer()
        trainer.train(epochs=5)
    except Exception as e:
        print(f"⚠️ Skin Cancer training failed: {e}")


def main():
    """Main deployment function"""
    print("=" * 60)
    print("🚀 Deploying Multi-Modal Medical Diagnosis Assistant")
    print("=" * 60)

    # Step 1: Create directories
    create_directories()

    # Step 2: Train models
    train_models()

    # Step 3: Launch application
    print("\n🎯 Launching Medical Diagnosis Assistant...")
    print("Application will be available at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                        "--server.port", "8501",
                        "--server.address", "localhost",
                        "--browser.gatherUsageStats", "false"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("Try running manually: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()