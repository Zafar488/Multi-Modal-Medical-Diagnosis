import re
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import requests

logger = logging.getLogger(__name__)

# Project root directory (one level up from app/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 132 symptoms the symptom model was trained on
SYMPTOM_FEATURES = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
    'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
    'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

EXTRACTION_SYSTEM_PROMPT = f"""
You are a medical parsing assistant. Your task is to analyze the conversation history and extract structured information.

Analyze the conversation history (paying special attention to the latest user message) and extract:
1. Symptoms: Identify any symptoms the user is experiencing. Map them ONLY to the following allowed symptoms list:
{SYMPTOM_FEATURES}
If a symptom is mentioned but not in this list, map it to the closest match or exclude it if no match exists.
2. Demographics & Context:
   - age: integer or null
   - gender: string ("male", "female", or null)
   - duration: string (description of how long symptoms lasted, or null)
   - severity: string (mild, moderate, severe, or null)
3. Missing Information: Determine if any of the following key details are missing:
   - "age"
   - "gender"
   - "duration"
   - "severity"
   Only list them as missing if the user has NOT provided them yet and they are relevant.
4. Out of Scope Check: Set "out_of_scope" to true if the latest user message is completely unrelated to medical, health, wellness, first aid, anatomy, or diagnostic queries (e.g. general programming, math, stories). Set to false otherwise.

You MUST respond in valid JSON format only, matching this structure:
{{
  "symptoms": ["cough", "high_fever"],
  "age": 25,
  "gender": "male",
  "duration": "3 days",
  "severity": "moderate",
  "missing_info": ["age", "gender"],
  "out_of_scope": false
}}
"""

class MedicalChatbot:
    """AI-powered medical chatbot with context management using Groq API"""
    
    def __init__(self, config_path=None, predictor=None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
        self.config = self.load_config(config_path)
        self.context = {
            'patient_name': None,
            'age': None,
            'gender': None,
            'symptoms': [],
            'current_diagnosis': None,
            'conversation_history': [],
            'last_interaction': datetime.now()
        }
        
        # Share or lazily import predictor to save memory
        self.predictor = predictor
        if self.predictor is None:
            try:
                from app.predictor import MedicalPredictor
                self.predictor = MedicalPredictor()
            except Exception as e:
                logger.error(f"Failed to load MedicalPredictor: {e}")
                self.predictor = None
        
        # Load Groq API Configuration
        chatbot_config = self.config.get('chatbot', {})
        self.groq_api_key = self._get_api_key()
        self.model = chatbot_config.get('model', 'llama-3.3-70b-versatile')
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        self.emergency_keywords = chatbot_config.get('emergency_keywords', [
            'emergency', 'urgent', 'severe pain', 'bleeding', 'unconscious',
            'heart attack', 'stroke', 'choking', 'suicide', 'kill myself'
        ])
        
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}
            
    def _get_api_key(self):
        """Robust loader for GROQ_API_KEY from environment, .env file, or config"""
        # 1. Environment variable
        api_key = os.environ.get('GROQ_API_KEY')
        if api_key:
            return api_key
            
        # 2. Manual parser for .env file
        env_path = os.path.join(PROJECT_ROOT, '.env')
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            if '=' in line:
                                key, val = line.split('=', 1)
                                if key.strip() == 'GROQ_API_KEY':
                                    return val.strip().strip('"').strip("'")
            except Exception as e:
                logger.warning(f"Error parsing .env file: {e}")
                
        # 3. Config YAML file
        chatbot_config = self.config.get('chatbot', {})
        api_key = chatbot_config.get('groq_api_key')
        if api_key:
            return api_key
            
        # 4. Fallback default
        return None

    def get_response(self, user_input: str, history: List[Tuple[str, str]] = None) -> str:
        """Get chatbot response using Groq LLM API with strict safety prompts"""
        user_input_clean = user_input.strip()
        
        # Sync conversation history from caller if provided (maintains isolation between Streamlit sessions)
        if history is not None:
            self.context['conversation_history'] = list(history)
        else:
            self.context['conversation_history'].append(('user', user_input_clean))
            
        self.context['last_interaction'] = datetime.now()
        
        # 1. Check for extreme emergency keywords (rule-based override)
        if any(keyword in user_input_clean.lower() for keyword in self.emergency_keywords):
            response = (
                "⚠️ **URGENT MEDICAL WARNING** ⚠️\n\n"
                "The symptoms or keywords you entered indicate a potential medical emergency.\n"
                "🆘 **Please seek immediate medical attention!**\n"
                "🚑 **Contact emergency services immediately (e.g., 911, 112, 102).**\n\n"
                "Do not wait or rely on this AI for emergency guidance."
            )
            self.context['conversation_history'].append(('assistant', response))
            return response
            
        # 2. Call Groq JSON extraction to understand symptoms and context
        extracted = self._extract_state_using_groq()
        
        # Update current context state
        if extracted:
            if extracted.get('age'):
                self.context['age'] = extracted['age']
            if extracted.get('gender'):
                self.context['gender'] = extracted['gender']
            if extracted.get('symptoms'):
                # Accumulate symptoms
                self.context['symptoms'] = list(set(self.context['symptoms'] + extracted['symptoms']))
                
            # Handle out of scope immediately
            if extracted.get('out_of_scope'):
                response = "I am MediAI, a medical assistant. I can only assist with medical, health, and symptom-related queries."
                self.context['conversation_history'].append(('assistant', response))
                return response
        else:
            extracted = {
                "symptoms": self.context['symptoms'],
                "age": self.context.get('age'),
                "gender": self.context.get('gender'),
                "duration": None,
                "severity": None,
                "missing_info": ["age", "gender", "duration", "severity"]
            }

        # 3. Choose path based on symptoms and details gathered
        history_messages = []
        for speaker, text in self.context['conversation_history'][-8:]:
            role = "user" if speaker == "user" else "assistant"
            history_messages.append({"role": role, "content": text})

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        # A. Case 1: No symptoms reported yet (e.g. general greeting or question)
        if not self.context['symptoms']:
            system_prompt = (
                "You are 'MediAI', a professional, empathetic, and objective medical AI assistant.\n"
                "Respond to the user's greeting or message in a caring manner, and politely ask them to describe their symptoms or health concerns.\n"
                "Keep the response brief (max 2-3 sentences)."
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}] + history_messages,
                "temperature": 0.5,
                "max_tokens": 200
            }
            return self._call_groq_api(headers, payload)

        # B. Case 2: Symptoms present, but key demographics/details are missing
        if extracted.get('missing_info') and len(self.context['symptoms']) > 0:
            # We want to gather the missing details (e.g., age, gender, duration, severity)
            missing_fields = [f.replace('_', ' ') for f in extracted['missing_info']]
            system_prompt = (
                "You are 'MediAI', a professional, empathetic medical assistant.\n"
                f"The user has reported symptoms ({self.context['symptoms']}), but you need to gather these missing details: {', '.join(missing_fields)}.\n"
                "Politely acknowledge what they've shared, and ask for the missing details in a warm, natural, and caring conversational tone.\n"
                "Keep it concise (max 3 sentences) and do not provide any diagnosis yet."
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}] + history_messages,
                "temperature": 0.4,
                "max_tokens": 250
            }
            return self._call_groq_api(headers, payload)

        # C. Case 3: We have symptoms and details — run prediction and explain it!
        try:
            # Construct symptom dict for local prediction
            symptoms_dict = {s: 0 for s in SYMPTOM_FEATURES}
            for s in self.context['symptoms']:
                if s in symptoms_dict:
                    symptoms_dict[s] = 1
                    
            if self.predictor:
                disease, confidence, all_probs = self.predictor.predict_symptoms(symptoms_dict)
                prediction_info = f"- Model Prediction: {disease} (Confidence: {confidence:.1%})"
            else:
                disease = "Unavailable (Model loading issue)"
                confidence = 0.0
                prediction_info = "- Model Prediction: Unavailable"
                
            system_prompt = (
                "You are 'MediAI', a professional, empathetic, and expert medical AI assistant.\n"
                "Based on the patient's symptoms and details, our machine learning prediction model has analyzed the case.\n\n"
                "Case Analysis Details:\n"
                f"- Reported Symptoms: {self.context['symptoms']}\n"
                f"- Age: {self.context.get('age')}\n"
                f"- Gender: {self.context.get('gender')}\n"
                f"- Duration: {extracted.get('duration') or 'Not specified'}\n"
                f"- Severity: {extracted.get('severity') or 'Moderate'}\n"
                f"{prediction_info}\n\n"
                "Provide a professional, clinical, and caring medical response that:\n"
                "1. Explains the model's prediction in a user-friendly, empathetic way.\n"
                "2. Categorizes the overall risk (Low, Medium, or High) based on symptom severity and the predicted condition.\n"
                "3. Lists clear, actionable health recommendations (lifestyle, over-the-counter remedies, monitoring advice).\n"
                "4. Reminds the patient clearly that this is a preliminary AI analysis and NOT a final clinical diagnosis, and they should consult a healthcare provider."
            )
            
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}] + history_messages,
                "temperature": 0.3,
                "max_tokens": 800
            }
            return self._call_groq_api(headers, payload)
            
        except Exception as e:
            logger.error(f"Error compiling prediction explanation: {e}")
            fallback = "I understand your concerns. However, I encountered an issue processing the diagnostics. Please consult a doctor."
            self.context['conversation_history'].append(('assistant', fallback))
            return fallback

    def _extract_state_using_groq(self) -> Optional[Dict]:
        """Calls Groq in JSON mode to extract symptoms, age, gender, duration, severity from history"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            # Use up to last 10 messages for state extraction context
            messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
            for speaker, text in self.context['conversation_history'][-10:]:
                role = "user" if speaker == "user" else "assistant"
                messages.append({"role": role, "content": text})
                
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "max_tokens": 400
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                result_json = response.json()
                content = result_json['choices'][0]['message']['content'].strip()
                return json.loads(content)
            else:
                logger.error(f"Groq API state extraction failed with code {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error extracting conversation state: {e}")
            return None

    def _call_groq_api(self, headers: Dict, payload: Dict) -> str:
        """Helper to post chat completions to Groq API with fallback"""
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                result_json = response.json()
                reply = result_json['choices'][0]['message']['content'].strip()
                self.context['conversation_history'].append(('assistant', reply))
                return reply
            else:
                logger.error(f"Groq API call returned error {response.status_code}: {response.text}")
                raise Exception("API status error")
        except Exception as e:
            logger.error(f"Error calling Groq API chat completion: {e}")
            fallback = "I understand your concern. However, I am currently experiencing connection difficulties. Please consult a medical professional."
            self.context['conversation_history'].append(('assistant', fallback))
            return fallback

    def reset_context(self):
        """Reset conversation context"""
        self.context = {
            'patient_name': None,
            'age': None,
            'gender': None,
            'symptoms': [],
            'current_diagnosis': None,
            'conversation_history': [],
            'last_interaction': datetime.now()
        }

if __name__ == "__main__":
    chatbot = MedicalChatbot()
    print("🤖 MediAI Chatbot: Type 'quit' to exit")
    print("-" * 50)
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("MediAI: Goodbye! Take care of your health!")
            break
        response = chatbot.get_response(user_input)
        print(f"\nMediAI: {response}")
        print("-" * 50)