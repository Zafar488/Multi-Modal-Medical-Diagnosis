"""
Chest X-Ray Pneumonia Detection Model Training
Uses MobileNetV2 transfer learning for fast, efficient training (max 5 epochs)
"""
import tensorflow as tf
import keras
from keras import layers, callbacks
from keras.applications import MobileNetV2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChestXRayTrainer:
    def __init__(self, img_size=224, batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['NORMAL', 'PNEUMONIA']

    def build_model(self, num_classes=2):
        """Build model using Transfer Learning with MobileNetV2"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        # Freeze base model for fast training
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

    def validate_images(self, data_dir):
        """Check for corrupted images and remove them"""
        removed = 0
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
            for class_name in self.class_names:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    try:
                        img = tf.io.read_file(fpath)
                        tf.image.decode_image(img)
                    except Exception:
                        logger.warning(f"🗑️ Removing corrupted image: {fpath}")
                        os.remove(fpath)
                        removed += 1
        logger.info(f"🧹 Image validation complete. Removed {removed} corrupted files.")

    def prepare_data(self, data_dir):
        """Prepare data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names
        )

        validation_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names
        )

        test_generator = None
        if os.path.exists(test_dir):
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )

        return train_generator, validation_generator, test_generator

    def train(self, data_dir=None, epochs=5):
        """Train the model"""
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, 'datasets', 'chest_xray')

        logger.info("🩻 Validating chest X-Ray images...")
        self.validate_images(data_dir)

        logger.info("🩻 Preparing Chest X-Ray data...")
        train_gen, val_gen, test_gen = self.prepare_data(data_dir)

        logger.info(f"📊 Classes: {train_gen.class_indices}")
        logger.info(f"📈 Training samples: {train_gen.samples}")
        logger.info(f"📉 Validation samples: {val_gen.samples}")

        self.model = self.build_model(len(train_gen.class_indices))

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

        logger.info(f"🚀 Training X-Ray model for {epochs} epochs...")
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
        model_path = os.path.join(models_dir, 'xray_model.h5')
        self.model.save(model_path)
        logger.info(f"✅ X-Ray model saved to {model_path}")

        # Evaluate on test set
        if test_gen:
            results = self.model.evaluate(test_gen, verbose=0)
            logger.info(f"\n📊 Test Results:")
            logger.info(f"  Loss: {results[0]:.4f}")
            logger.info(f"  Accuracy: {results[1]:.4f}")

        return self.model


if __name__ == "__main__":
    trainer = ChestXRayTrainer()
    trainer.train(epochs=5)