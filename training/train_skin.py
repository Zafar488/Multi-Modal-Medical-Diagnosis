"""
Skin Cancer Detection Model Training
Uses HAM10000 dataset with MobileNetV2 transfer learning (max 5 epochs)
Maps dx codes to binary: mel → malignant, rest → benign
"""
import tensorflow as tf
import keras
from keras import layers, callbacks
from keras.applications import MobileNetV2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import numpy as np
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HAM10000 diagnosis codes → binary mapping
# mel (Melanoma) → malignant
# Everything else (nv, bkl, bcc, akiec, vasc, df) → benign
MALIGNANT_CODES = {'mel', 'bcc'}  # melanoma + basal cell carcinoma
BENIGN_CODES = {'nv', 'bkl', 'akiec', 'vasc', 'df'}


class SkinCancerTrainer:
    def __init__(self, img_size=224, batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['benign', 'malignant']

    def build_model(self, num_classes=2):
        """Build model using Transfer Learning with MobileNetV2"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def prepare_data(self):
        """Prepare data from HAM10000 metadata + image folders"""
        skin_dir = os.path.join(PROJECT_ROOT, 'datasets', 'skin_cancer')
        metadata_path = os.path.join(skin_dir, 'HAM10000_metadata.csv')
        img_dir_1 = os.path.join(skin_dir, 'HAM10000_images_part_1')
        img_dir_2 = os.path.join(skin_dir, 'HAM10000_images_part_2')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        # Load metadata
        df = pd.read_csv(metadata_path)
        logger.info(f"📥 Loaded metadata: {len(df)} records")
        logger.info(f"📊 Diagnosis distribution:\n{df['dx'].value_counts().to_string()}")

        # Map dx to binary label
        df['label'] = df['dx'].apply(
            lambda x: 'malignant' if x in MALIGNANT_CODES else 'benign'
        )
        logger.info(f"📊 Binary label distribution:\n{df['label'].value_counts().to_string()}")

        # Build full image path for each record
        def find_image_path(image_id):
            for img_dir in [img_dir_1, img_dir_2]:
                path = os.path.join(img_dir, f"{image_id}.jpg")
                if os.path.exists(path):
                    return path
            return None

        df['filepath'] = df['image_id'].apply(find_image_path)

        # Remove records with missing images
        missing = df['filepath'].isna().sum()
        if missing > 0:
            logger.warning(f"⚠️ {missing} images not found, removing from dataset")
            df = df.dropna(subset=['filepath'])

        # Remove duplicates by lesion_id — keep first image per lesion
        before = len(df)
        df = df.drop_duplicates(subset='lesion_id', keep='first')
        logger.info(f"🧹 Deduplicated by lesion_id: {before} → {len(df)} images")

        # Validate images — check each can be read
        valid_rows = []
        for idx, row in df.iterrows():
            try:
                raw = tf.io.read_file(row['filepath'])
                tf.image.decode_jpeg(raw, channels=3)
                valid_rows.append(idx)
            except Exception:
                logger.warning(f"🗑️ Corrupted image: {row['filepath']}")

        df = df.loc[valid_rows]
        logger.info(f"✅ Valid images after corruption check: {len(df)}")

        # Split: 80% train, 20% validation
        from sklearn.model_selection import train_test_split
        df_train, df_val = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['label']
        )

        logger.info(f"📈 Training samples: {len(df_train)}")
        logger.info(f"📉 Validation samples: {len(df_val)}")

        # Create data generators using flow_from_dataframe
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col='filepath',
            y_col='label',
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=True
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col='filepath',
            y_col='label',
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )

        return train_generator, val_generator

    def train(self, epochs=5):
        """Train the model"""
        logger.info("🩺 Preparing Skin Cancer data from HAM10000...")
        train_gen, val_gen = self.prepare_data()

        logger.info(f"📊 Classes: {train_gen.class_indices}")

        self.model = self.build_model(num_classes=2)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary(print_fn=logger.info)

        # Ensure models directory exists
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)

        callbacks_list = [
            callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
        ]

        logger.info(f"🚀 Training Skin Cancer model for {epochs} epochs...")
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.samples // self.batch_size),
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=max(1, val_gen.samples // self.batch_size),
            callbacks=callbacks_list,
            verbose=1
        )

        # Save final model
        model_path = os.path.join(models_dir, 'skin_model.h5')
        self.model.save(model_path)
        logger.info(f"✅ Skin Cancer model saved to {model_path}")

        # Print final metrics
        val_results = self.model.evaluate(val_gen, verbose=0)
        logger.info(f"\n📊 Final Validation Results:")
        logger.info(f"  Loss: {val_results[0]:.4f}")
        logger.info(f"  Accuracy: {val_results[1]:.4f}")

        return self.model


if __name__ == "__main__":
    trainer = SkinCancerTrainer()
    trainer.train(epochs=5)