import os
import base64
from io import BytesIO

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse

from models.preprocessing import detect_and_crop_face
from models.inference import predict_skin_type, predict_acne_level, predict_other_concerns
from models.skin_tone.skin_tone_knn import identify_skin_tone
from models.recommender.rec import recs_essentials, makeup_recommendation, features as FEATURE_NAMES

app = Flask(__name__)
CORS(app)
api = Api(app)

SKIN_TONE_DATASET = "models/skin_tone/skin_tone_dataset.csv"
STATIC_DIR = "./static"
FACE_IMAGE = os.path.join(STATIC_DIR, "face.png")
UPLOAD_IMAGE = os.path.join(STATIC_DIR, "upload.png")

model1 = None
model2 = None
skinmate_model = None


def load_models():
    global model1, model2, skinmate_model
    model1 = load_model("./models/skin_model")
    print("Skin type model loaded")
    model2 = load_model("./models/acne_model")
    print("Acne model loaded")
    skinmate_model = tf.saved_model.load("./models/other_concerns")
    print("Other concerns model loaded")


def save_uploaded_image(file_data: str) -> str:
    starter = file_data.find(",")
    image_data = file_data[starter + 1 :]
    image_bytes = base64.b64decode(bytes(image_data, encoding="ascii"))
    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    im.thumbnail((1024, 1024), resample)
    os.makedirs(STATIC_DIR, exist_ok=True)
    im.save(UPLOAD_IMAGE)
    return UPLOAD_IMAGE


def analyze_face_image(face_path: str) -> dict:
    skin = predict_skin_type(model1, face_path)
    acne = predict_acne_level(model2, face_path)
    tone_result = identify_skin_tone(face_path, dataset=SKIN_TONE_DATASET)
    other = predict_other_concerns(skinmate_model, face_path)

    return {
        "type": skin["display_type"],
        "tone": str(tone_result["tone"]),
        "acne": acne["label"],
        "other_concerns": other,
        "confidence": {
            "skin_type": skin,
            "acne": acne,
            "skin_tone": tone_result,
        },
    }


load_models()

img_put_args = reqparse.RequestParser()
img_put_args.add_argument(
    "file", help="Please provide a valid image file", required=True, location="json"
)

rec_args = reqparse.RequestParser()
rec_args.add_argument("tone", type=int, help="Argument required", required=True, location="json")
rec_args.add_argument("type", type=str, help="Argument required", required=True, location="json")
rec_args.add_argument("features", type=dict, help="Argument required", required=True, location="json")
rec_args.add_argument("acne", type=str, required=False, location="json")


def build_feature_vector(features):
    return [int(features.get(name, 0)) for name in FEATURE_NAMES]


class Recommendation(Resource):
    def put(self):
        args = rec_args.parse_args()
        features = args["features"]
        tone = args["tone"]
        skin_type = args["type"].lower()
        acne_level = (args.get("acne") or "Moderate").lower()

        if skin_type == "all":
            skin_type = "normal"

        skin_tone_bucket = "light to medium"
        if tone <= 2:
            skin_tone_bucket = "fair to light"
        elif tone >= 4:
            skin_tone_bucket = "medium to dark"

        # Auto-enable acne concern for moderate/severe predictions
        if acne_level in ("moderate", "severe"):
            features["acne"] = 1

        fv = build_feature_vector(features)
        general = recs_essentials(vector=fv, skin_type=skin_type)
        makeup = makeup_recommendation(skin_tone_bucket, skin_type)
        return {"general": general, "makeup": makeup}


class SkinMetrics(Resource):
    def put(self):
        args = img_put_args.parse_args()
        try:
            upload_path = save_uploaded_image(args["file"])
            crop_meta = detect_and_crop_face(upload_path, FACE_IMAGE)

            if not crop_meta.get("face_detected"):
                return {"error": crop_meta.get("message", "Could not process image.")}, 400

            result = analyze_face_image(FACE_IMAGE)
            result["face_detection"] = crop_meta
            if crop_meta.get("used_fallback"):
                result["warning"] = crop_meta.get("message")
            return result, 200
        except Exception as exc:
            return {"error": str(exc)}, 500


api.add_resource(SkinMetrics, "/upload")
api.add_resource(Recommendation, "/recommend")


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": model1 is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
