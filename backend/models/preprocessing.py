"""
Face detection, cropping, and lighting normalization for skin analysis.
"""
import os

import cv2
import numpy as np

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _apply_clahe(bgr_image: np.ndarray) -> np.ndarray:
    """Normalize lighting using CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _detect_faces(gray: np.ndarray):
    """Try progressively relaxed Haar settings before giving up."""
    configs = [
        {"scaleFactor": 1.08, "minNeighbors": 5, "minSize": (80, 80)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (60, 60)},
        {"scaleFactor": 1.03, "minNeighbors": 2, "minSize": (40, 40)},
    ]
    for cfg in configs:
        faces = _FACE_CASCADE.detectMultiScale(
            gray, flags=cv2.CASCADE_SCALE_IMAGE, **cfg
        )
        if len(faces) > 0:
            return faces
    return []


def _center_crop_fallback(image: np.ndarray, fraction: float = 0.75) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h = int(height * fraction)
    crop_w = int(width * fraction)
    y1 = max(0, (height - crop_h) // 2)
    x1 = max(0, (width - crop_w) // 2)
    return image[y1 : y1 + crop_h, x1 : x1 + crop_w]


def detect_and_crop_face(
    image_path: str,
    output_path: str,
    padding_ratio: float = 0.25,
    min_face_fraction: float = 0.08,
) -> dict:
    """
    Detect the largest frontal face, crop with padding, normalize lighting, and save.

    Returns metadata including whether a face was found and the crop box used.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file.")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces(gray)

    if len(faces) == 0:
        fallback = _center_crop_fallback(image)
        fallback = _apply_clahe(fallback)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, fallback)
        return {
            "face_detected": True,
            "used_fallback": True,
            "message": "No face detected — using center crop. Use a clear front-facing photo for best accuracy.",
            "face_box": None,
            "face_fraction": 0.0,
            "output_path": output_path,
        }

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_area = w * h
    image_area = width * height
    if face_area / image_area < min_face_fraction:
        fallback = _center_crop_fallback(image)
        fallback = _apply_clahe(fallback)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, fallback)
        return {
            "face_detected": True,
            "used_fallback": True,
            "message": "Face was too small — using center crop. Move closer for best accuracy.",
            "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "face_fraction": round(face_area / image_area, 3),
            "output_path": output_path,
        }

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return {
            "face_detected": False,
            "message": "Could not extract a valid face region from the image.",
        }

    face_crop = _apply_clahe(face_crop)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, face_crop)

    return {
        "face_detected": True,
        "used_fallback": False,
        "face_box": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
        "face_fraction": round(face_area / image_area, 3),
        "output_path": output_path,
    }
