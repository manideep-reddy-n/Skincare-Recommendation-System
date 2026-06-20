"""
Fitzpatrick skin tone classification using KNN on segmented skin color features.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.skin_tone.skin_detection import skin_detection

_CLASSIFIER = None
_DATASET_PATH = None


def _get_classifier(dataset_path: str) -> Pipeline:
    global _CLASSIFIER, _DATASET_PATH
    if _CLASSIFIER is not None and _DATASET_PATH == dataset_path:
        return _CLASSIFIER

    df = pd.read_csv(dataset_path)
    X = df.iloc[:, [1, 2, 3]].values
    y = df.iloc[:, 0].values

    _CLASSIFIER = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )
    _CLASSIFIER.fit(X, y)
    _DATASET_PATH = dataset_path
    return _CLASSIFIER


def identify_skin_tone(image_path, dataset):
    mean_color_values = skin_detection(image_path)
    classifier = _get_classifier(dataset)
    tone = int(classifier.predict([mean_color_values])[0])
    probs = classifier.predict_proba([mean_color_values])[0]
    classes = classifier.named_steps["knn"].classes_
    confidence = float(np.max(probs))
    return {
        "tone": tone,
        "confidence": round(confidence, 3),
        "probabilities": {
            str(int(classes[i])): round(float(probs[i]), 3) for i in range(len(classes))
        },
    }
