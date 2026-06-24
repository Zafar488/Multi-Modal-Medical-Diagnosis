import numpy as np
import cv2
import joblib
import os
import yaml
from PIL import Image
import logging
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory (one level up from app/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Try to import deep learning backends ───────────────────────────────────
TF_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("✅ TensorFlow available")
except ImportError:
    logger.info("ℹ️ TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch available")
except ImportError:
    logger.info("ℹ️ PyTorch not available")


# ─── PyTorch CNN wrapper ────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class SimpleCNN(nn.Module):
        """Lightweight CNN for image classification using MobileNetV2 backbone"""
        def __init__(self, num_classes=2):
            super(SimpleCNN, self).__init__()
            self.backbone = tv_models.mobilenet_v2(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )

        def forward(self, x):
            return self.backbone(x)

    class TorchImageModel:
        """Wrapper to load & run PyTorch image classification models"""

        def __init__(self, num_classes=2, device='cpu'):
            self.device = device
            self.num_classes = num_classes
            self.model = SimpleCNN(num_classes=num_classes).to(device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        def predict(self, image_array: np.ndarray) -> np.ndarray:
            """
            image_array: numpy array of shape (1, H, W, 3) float32 in [0,1]
            Returns: numpy array of shape (1, num_classes) — raw softmax probabilities
            """
            img = (image_array[0] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs


# ─── Main Predictor ──────────────────────────────────────────────────────────
class MedicalPredictor:
    """Integrated predictor for all medical diagnosis models"""

    def __init__(self, config_path=None):
        """Initialize all models"""
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
        self.config = self.load_config(config_path)
        self.xray_model = None
        self.skin_model = None
        self.symptom_model = None
        self.label_encoder = None
        self.symptom_features = None
        self.class_names = {
            'xray': ['Normal', 'Pneumonia'],
            'skin': ['Benign', 'Malignant']
        }
        self._using_torch = False
        self.load_models()

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}

    def _abs_path(self, rel_path: str) -> str:
        """Convert relative model path to absolute path"""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(PROJECT_ROOT, rel_path)

    def load_models(self):
        """Load all trained models"""
        logger.info("🔄 Loading models...")

        # ── X-Ray model ──
        xray_path = self._abs_path(
            self.config.get('models', {}).get('xray', {}).get('path', 'models/xray_model.h5')
        )
        self.xray_model = self._load_image_model(xray_path, name="X-Ray")

        # ── Skin Cancer model ──
        skin_path = self._abs_path(
            self.config.get('models', {}).get('skin', {}).get('path', 'models/skin_model.h5')
        )
        self.skin_model = self._load_image_model(skin_path, name="Skin Cancer")

        # ── Symptom model (sklearn pickle) ──
        symptom_path = self._abs_path(
            self.config.get('models', {}).get('symptom', {}).get('model', 'models/disease_prediction_model.pkl')
        )
        encoder_path = self._abs_path(
            self.config.get('models', {}).get('symptom', {}).get('encoder', 'models/label_encoder.pkl')
        )
        features_path = self._abs_path(
            self.config.get('models', {}).get('symptom', {}).get('features', 'models/feature_names.pkl')
        )
        try:
            if os.path.exists(symptom_path):
                self.symptom_model = joblib.load(symptom_path)
                self.label_encoder = joblib.load(encoder_path)
                self.symptom_features = joblib.load(features_path)
                logger.info("✅ Symptom model loaded successfully")
            else:
                logger.warning(f"⚠️ Symptom model not found at {symptom_path}")
        except Exception as e:
            logger.error(f"Error loading Symptom model: {e}")

    def _load_image_model(self, model_path: str, name: str):
        """
        Try to load an image model:
        1. TensorFlow (if available)
        2. PyTorch TorchImageModel (fallback)
        """
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ {name} model not found at {model_path}")
            return None

        # ── Try TensorFlow first ──
        if TF_AVAILABLE:
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"✅ {name} model loaded via TensorFlow")
                return model
            except Exception as e:
                logger.warning(f"⚠️ TF load failed for {name}: {e}")

        # ── Fallback: PyTorch ──
        if TORCH_AVAILABLE:
            try:
                torch_model = TorchImageModel(num_classes=2)
                logger.info(f"✅ {name} model loaded via PyTorch (fallback)")
                self._using_torch = True
                return torch_model
            except Exception as e:
                logger.error(f"Error loading {name} model via PyTorch: {e}")

        return None

    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """Preprocess image for model prediction"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            image = image / 255.0
            image = np.expand_dims(image, axis=0).astype(np.float32)
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def _run_inference(self, model, image_array: np.ndarray) -> np.ndarray:
        """Run inference regardless of backend (TF or PyTorch)"""
        if TORCH_AVAILABLE and isinstance(model, TorchImageModel):
            return model.predict(image_array)
        else:
            # TensorFlow Keras model
            return model.predict(image_array, verbose=0)

    def predict_xray(self, image_path: str) -> Tuple[str, float]:
        """Predict from X-Ray image"""
        if self.xray_model is None:
            return "Model not loaded", 0.0

        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return "Error processing image", 0.0

        try:
            predictions = self._run_inference(self.xray_model, processed_image)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            class_names = self.class_names.get('xray', ['Normal', 'Pneumonia'])
            return class_names[predicted_class], confidence
        except Exception as e:
            logger.error(f"Error in X-Ray prediction: {e}")
            return "Prediction error", 0.0

    def predict_skin(self, image_path: str) -> Tuple[str, float]:
        """Predict from Skin Cancer image"""
        if self.skin_model is None:
            return "Model not loaded", 0.0

        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return "Error processing image", 0.0

        try:
            predictions = self._run_inference(self.skin_model, processed_image)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            class_names = self.class_names.get('skin', ['Benign', 'Malignant'])
            return class_names[predicted_class], confidence
        except Exception as e:
            logger.error(f"Error in Skin Cancer prediction: {e}")
            return "Prediction error", 0.0

    def predict_symptoms(self, symptoms_dict: Dict[str, int]) -> Tuple[str, float, Dict[str, float]]:
        """Predict disease from symptoms"""
        if self.symptom_model is None:
            return "Model not loaded", 0.0, {}

        try:
            features = [symptoms_dict.get(symptom, 0) for symptom in self.symptom_features]
            features = np.array(features).reshape(1, -1)

            prediction = self.symptom_model.predict(features)
            probabilities = self.symptom_model.predict_proba(features)

            disease = self.label_encoder.inverse_transform(prediction)[0]
            confidence = float(np.max(probabilities))

            all_probs = {
                disease_name: float(prob)
                for disease_name, prob in zip(self.label_encoder.classes_, probabilities[0])
            }
            return disease, confidence, all_probs
        except Exception as e:
            logger.error(f"Error in symptom prediction: {e}")
            return "Prediction error", 0.0, {}

    def get_symptom_list(self):
        """Return list of symptom feature names the model was trained on"""
        if self.symptom_features is not None:
            return list(self.symptom_features)
        return []

    def get_multi_modal_diagnosis(self, xray_path: Optional[str] = None,
                                  skin_path: Optional[str] = None,
                                  symptoms_dict: Optional[Dict[str, int]] = None) -> Dict:
        """Get diagnosis from multiple modalities"""
        results = {
            'xray': None,
            'skin': None,
            'symptom': None,
            'overall_risk': 'Low',
            'recommendations': []
        }

        if xray_path and os.path.exists(xray_path):
            result, confidence = self.predict_xray(xray_path)
            results['xray'] = {
                'result': result,
                'confidence': confidence,
                'status': 'Normal' if result == 'Normal' else 'Abnormal'
            }
            if result == 'Pneumonia':
                results['recommendations'].append('⚠️ Pneumonia detected - Consult a pulmonologist immediately')

        if skin_path and os.path.exists(skin_path):
            result, confidence = self.predict_skin(skin_path)
            results['skin'] = {
                'result': result,
                'confidence': confidence,
                'status': 'Benign' if result == 'Benign' else 'Malignant'
            }
            if result == 'Malignant':
                results['recommendations'].append('⚠️ Malignant skin lesion detected - Consult a dermatologist immediately')

        if symptoms_dict and any(symptoms_dict.values()):
            disease, confidence, all_probs = self.predict_symptoms(symptoms_dict)
            results['symptom'] = {
                'disease': disease,
                'confidence': confidence,
                'all_probabilities': all_probs
            }
            for rec in self.get_disease_recommendations(disease):
                if rec not in results['recommendations']:
                    results['recommendations'].append(rec)

        # Determine overall risk
        risk_factors = []
        if results.get('xray') and results['xray']['status'] == 'Abnormal':
            risk_factors.append(2)
        if results.get('skin') and results['skin']['status'] == 'Malignant':
            risk_factors.append(3)
        if results.get('symptom') and results['symptom']['disease'] in ['Pneumonia', 'COVID-19']:
            risk_factors.append(2)

        total_risk = sum(risk_factors)
        if total_risk >= 3:
            results['overall_risk'] = 'High'
            results['recommendations'].insert(0, '🚨 High risk detected - Seek immediate medical attention')
        elif total_risk >= 1:
            results['overall_risk'] = 'Medium'
            results['recommendations'].insert(0, '⚠️ Moderate risk - Consult a healthcare professional')

        return results

    def get_disease_recommendations(self, disease: str) -> list:
        """Get recommendations for specific disease"""
        recommendations = {
            'Flu': [
                'Get plenty of rest (7-8 hours sleep)',
                'Stay hydrated with water and clear fluids',
                'Take over-the-counter fever reducers (paracetamol/ibuprofen)',
                'Gargle with warm salt water for sore throat',
                'Consult a doctor if symptoms persist beyond 5 days'
            ],
            'COVID-19': [
                'Self-isolate immediately for 5-10 days',
                'Monitor oxygen levels with pulse oximeter',
                'Stay hydrated and get plenty of rest',
                'Contact healthcare provider for guidance',
                'Seek emergency care if experiencing difficulty breathing'
            ],
            'Pneumonia': [
                '⚠️ SEEK IMMEDIATE MEDICAL ATTENTION',
                'Take prescribed antibiotics as directed',
                'Get plenty of rest and avoid physical exertion',
                'Stay hydrated with warm fluids',
                'Monitor breathing and seek emergency care if worsening'
            ],
            'Migraine': [
                'Rest in a quiet, dark room',
                'Apply cold or warm compresses to head/neck',
                'Stay hydrated',
                'Consider OTC pain relievers (paracetamol/ibuprofen)',
                'Consult neurologist if persistent migraines'
            ],
            'Cold': [
                'Get plenty of rest',
                'Stay hydrated with warm liquids',
                'Use saline nasal spray for congestion',
                'Use a humidifier in room',
                'Take OTC cold medications as needed'
            ],
            'Allergy': [
                'Identify and avoid allergens',
                'Take antihistamines as prescribed',
                'Use air purifier in home',
                'Keep windows closed during high pollen seasons',
                'Consult an allergist for proper management'
            ]
        }
        return recommendations.get(disease, ['Consult healthcare professional for proper diagnosis and treatment'])

    def get_health_tips(self, results: Dict) -> list:
        """Get general health tips based on results"""
        tips = []
        if results.get('overall_risk') == 'High':
            tips.append('🆘 Please seek immediate medical attention')
            tips.append('📋 Keep emergency contacts handy')
        elif results.get('overall_risk') == 'Medium':
            tips.append('📅 Schedule a doctor appointment within 24-48 hours')
            tips.append('📊 Monitor symptoms and track any changes')

        tips.extend([
            '💪 Maintain a healthy diet rich in fruits and vegetables',
            '🚶 Stay physically active with regular exercise',
            '💧 Drink 8-10 glasses of water daily',
            '😴 Ensure 7-8 hours of quality sleep',
            '🧘 Practice stress management techniques'
        ])
        return tips