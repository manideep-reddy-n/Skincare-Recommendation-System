"""
Centralized model inference with test-time augmentation and confidence scores.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Training folders were sorted alphabetically:
# dry_skin, normal_skin, oil_skin
SKIN_TYPE_CLASSES = ["Dry", "Normal", "Oil"]
SKIN_TYPE_LABELS = ["Dry_skin", "Normal_skin", "Oil_skin"]

# Training folders: Level_0, Level_1, Level_2
ACNE_CLASSES = ["Low", "Moderate", "Severe"]

CONCERN_KEYS = ["Acnes", "Blackheads", "Darkspots", "Wrinkles"]
MIN_CONFIDENCE = 0.40


def _load_img_tensor(img_path: str, size: tuple) -> np.ndarray:
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr / 255.0


def _predict_with_tta(model, img_path: str, size=(224, 224)) -> tuple:
    """Average predictions on original and horizontally flipped image."""
    tensor = _load_img_tensor(img_path, size)
    flipped = np.flip(tensor, axis=2)
    batch = np.concatenate([tensor, flipped], axis=0)
    preds = model.predict(batch, verbose=0)
    return np.mean(preds, axis=0)


def _format_prediction(probs: np.ndarray, class_names: list) -> dict:
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    return {
        "label": class_names[idx],
        "confidence": round(confidence, 3),
        "probabilities": {
            class_names[i]: round(float(probs[i]), 3) for i in range(len(class_names))
        },
        "low_confidence": confidence < MIN_CONFIDENCE,
    }


def predict_skin_type(model, img_path: str) -> dict:
    probs = _predict_with_tta(model, img_path, size=(224, 224))
    result = _format_prediction(probs, SKIN_TYPE_CLASSES)
    result["display_type"] = "Oily" if result["label"] == "Oil" else result["label"]
    return result


def predict_acne_level(model, img_path: str) -> dict:
    probs = _predict_with_tta(model, img_path, size=(224, 224))
    return _format_prediction(probs, ACNE_CLASSES)


def predict_other_concerns(skinmate_model, img_path: str) -> dict:
    """
    Run the SkinMate multi-output model using the original calibrated scaling.
    """
    tensor = _load_img_tensor(img_path, size=(150, 150))
    img_tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    infer = skinmate_model.signatures["serving_default"]
    raw = list(infer(tf.constant(img_tensor)).values())[0].numpy().flatten()

    preds_normalized = raw * 1000
    preds_rounded = np.round(preds_normalized)
    preds_in_level = preds_rounded / 100
    preds_in_level_rounded = np.ceil(preds_in_level).flatten().astype(int)

    results = {
        CONCERN_KEYS[i]: int(np.clip(preds_in_level_rounded[i], 0, 10))
        for i in range(len(CONCERN_KEYS))
    }
    max_index = int(np.argmax(preds_in_level_rounded))
    results["Most_significant_problem"] = CONCERN_KEYS[max_index]
    results["severity_scores"] = [results[k] for k in CONCERN_KEYS]
    return results
