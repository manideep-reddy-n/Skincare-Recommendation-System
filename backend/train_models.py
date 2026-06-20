"""
Improved training script for skin type / acne CNN models.

Usage (from backend/ with venv active):
  python train_models.py --task skin_type --data-dir path/to/images
  python train_models.py --task acne --data-dir path/to/images

Expected folder layout:
  data-dir/
    train/
      class_a/
      class_b/
    val/
      class_a/
      class_b/

Requires at least ~50 images per class for reasonable results.
"""
import argparse
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EFFICIENTNET_URL = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def build_model(num_classes: int, fine_tune_at: int = 150) -> tf.keras.Model:
    base = hub.KerasLayer(
        EFFICIENTNET_URL,
        trainable=True,
        name="efficientnet_b0",
        input_shape=IMAGE_SIZE + (3,),
    )
    # Freeze early layers, fine-tune top blocks for better accuracy
    if hasattr(base, "trainable_weights"):
        for i, layer in enumerate(base.trainable_weights):
            layer._trainable = i >= fine_tune_at

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = base(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_generators(data_dir: str):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["skin_type", "acne"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train_data, val_data = get_generators(args.data_dir)
    num_classes = train_data.num_classes
    print(f"Classes: {train_data.class_indices}")

    model = build_model(num_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    output = args.output or f"./models/{args.task}_retrained"
    model.save(output)
    print(f"Saved model to {output}")
    print(f"Best val accuracy: {max(history.history.get('val_accuracy', [0])):.3f}")


if __name__ == "__main__":
    main()
