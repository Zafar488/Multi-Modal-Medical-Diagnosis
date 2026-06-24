"""
Symptom-Based Disease Prediction Model Training
Uses the real Training.csv / Testing.csv datasets (132 symptoms, 41 diseases)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SymptomTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None

    def load_and_clean_data(self):
        """Load and clean the real symptom datasets"""
        train_path = os.path.join(PROJECT_ROOT, 'datasets', 'symptoms', 'Training.csv')
        test_path = os.path.join(PROJECT_ROOT, 'datasets', 'symptoms', 'Testing.csv')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset not found at {train_path}")

        # Load datasets
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None

        logger.info(f"📥 Loaded training data: {df_train.shape}")
        if df_test is not None:
            logger.info(f"📥 Loaded testing data: {df_test.shape}")

        # The last column is 'prognosis' (the disease label)
        # Clean column names — remove trailing spaces/commas
        df_train.columns = [col.strip().rstrip(',') for col in df_train.columns]
        if df_test is not None:
            df_test.columns = [col.strip().rstrip(',') for col in df_test.columns]

        # Drop any fully empty columns
        df_train = df_train.dropna(axis=1, how='all')
        if df_test is not None:
            df_test = df_test.dropna(axis=1, how='all')

        # Handle missing values — fill NaN with 0 (symptom not present)
        df_train = df_train.fillna(0)
        if df_test is not None:
            df_test = df_test.fillna(0)

        # Remove duplicate rows
        before = len(df_train)
        df_train = df_train.drop_duplicates()
        dropped = before - len(df_train)
        if dropped > 0:
            logger.info(f"🧹 Removed {dropped} duplicate rows from training data")

        # Clean the prognosis column — strip whitespace
        df_train['prognosis'] = df_train['prognosis'].str.strip()
        if df_test is not None:
            df_test['prognosis'] = df_test['prognosis'].str.strip()

        logger.info(f"📊 Diseases found: {df_train['prognosis'].nunique()}")
        logger.info(f"📋 Symptoms (features): {len(df_train.columns) - 1}")

        return df_train, df_test

    def train(self):
        """Train symptom prediction model using real data"""
        logger.info("🤒 Preparing Symptom dataset...")

        df_train, df_test = self.load_and_clean_data()

        # Separate features and labels
        self.feature_names = [col for col in df_train.columns if col != 'prognosis']

        X_train = df_train[self.feature_names].values.astype(np.float32)
        y_train_raw = df_train['prognosis'].values

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train_raw)

        logger.info(f"📈 Training samples: {len(X_train)}")
        logger.info(f"📊 Unique diseases: {len(self.label_encoder.classes_)}")
        logger.info(f"📋 Feature count: {len(self.feature_names)}")

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        logger.info("🚀 Training Symptom model...")
        self.model.fit(X_train, y_train)

        # Evaluate on training data
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        logger.info(f"📊 Training Accuracy: {train_acc:.4f}")

        # Evaluate on test data if available
        if df_test is not None and len(df_test) > 0:
            # Ensure test has the same columns
            for col in self.feature_names:
                if col not in df_test.columns:
                    df_test[col] = 0

            X_test = df_test[self.feature_names].values.astype(np.float32)
            y_test_raw = df_test['prognosis'].values

            # Only evaluate on diseases that exist in training
            valid_mask = np.isin(y_test_raw, self.label_encoder.classes_)
            if valid_mask.sum() > 0:
                X_test = X_test[valid_mask]
                y_test_raw = y_test_raw[valid_mask]
                y_test = self.label_encoder.transform(y_test_raw)

                y_test_pred = self.model.predict(X_test)
                test_acc = accuracy_score(y_test, y_test_pred)
                logger.info(f"📊 Test Accuracy: {test_acc:.4f}")
                logger.info("\nClassification Report (Test):")
                logger.info("\n" + classification_report(
                    y_test, y_test_pred,
                    target_names=self.label_encoder.classes_,
                    zero_division=0
                ))

        # Save model, encoder, and feature names
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, 'disease_prediction_model.pkl')
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        features_path = os.path.join(models_dir, 'feature_names.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        joblib.dump(self.feature_names, features_path)

        logger.info(f"✅ Symptom model saved to {model_path}")
        logger.info(f"✅ Label encoder saved to {encoder_path}")
        logger.info(f"✅ Feature names saved to {features_path}")

        return self.model


if __name__ == "__main__":
    trainer = SymptomTrainer()
    trainer.train()