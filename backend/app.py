from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =============================
# Load trained model
# =============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# =============================
# Class names (alphabetical order of dataset folders — MUST match training)
# =============================
CLASS_META = {
    "Pepper__bell___Bacterial_spot":               ("Pepper", "Bacterial_spot"),
    "Pepper__bell___healthy":                       ("Pepper", "healthy"),
    "Potato___Early_blight":                        ("Potato", "Early_blight"),
    "Potato___Late_blight":                         ("Potato", "Late_blight"),
    "Potato___healthy":                             ("Potato", "healthy"),
    "Tomato_Bacterial_spot":                        ("Tomato", "Bacterial_spot"),
    "Tomato_Early_blight":                          ("Tomato", "Early_blight"),
    "Tomato_Late_blight":                           ("Tomato", "Late_blight"),
    "Tomato_Leaf_Mold":                             ("Tomato", "Leaf_Mold"),
    "Tomato_Septoria_leaf_spot":                    ("Tomato", "Septoria_leaf_spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite":  ("Tomato", "Spider_mites"),
    "Tomato__Target_Spot":                          ("Tomato", "Target_Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":        ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"),
    "Tomato__Tomato_mosaic_virus":                  ("Tomato", "Tomato_mosaic_virus"),
    "Tomato_healthy":                               ("Tomato", "healthy"),
}
class_names = list(CLASS_META.keys())

# =============================
# Advice dictionary
# =============================
ADVICE = {
    "Bacterial_spot": "Use copper-based fungicide and avoid overhead watering.",
    "Early_blight": "Remove infected leaves and use fungicide.",
    "Late_blight": "Destroy infected plants and avoid wet conditions.",
    "Leaf_Mold": "Improve air circulation and reduce humidity.",
    "Septoria_leaf_spot": "Remove infected leaves and apply fungicide.",
    "Spider_mites": "Use insecticidal soap or neem oil.",
    "Target_Spot": "Use fungicide and avoid wet leaves.",
    "Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants and control whiteflies.",
    "Bacterial": "Use recommended pesticide.",
}

# =============================
# Utility: preprocess image
# =============================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =============================
# Utility: severity from confidence
# =============================
def get_severity(conf):
    if conf > 0.75:
        return "Severe"
    elif conf > 0.4:
        return "Moderate"
    else:
        return "Mild"

# =============================
# API: Test
# =============================
@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({"status": "Backend working!"})

# =============================
# API: Predict
# =============================
@app.route("/api/crop-diagnosis", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    img_array = preprocess_image(img)

    pred = model.predict(img_array)

    # ✅ Proper softmax
    probs = np.exp(pred[0]) / np.sum(np.exp(pred[0]))

    predicted_index = int(np.argmax(probs))
    confidence = float(probs[predicted_index])

    label = class_names[predicted_index]

    # Parse label using metadata map
    crop, disease_raw = CLASS_META[label]

    if disease_raw.lower() == "healthy":
        status = "Healthy"
        disease = "None"
        severity = "None"
        advice = "Your crop looks healthy!"
    else:
        status = "Diseased"
        disease = disease_raw.replace("_", " ")
        severity = get_severity(confidence)

        # Find advice
        advice = "Remove infected leaves and avoid excess watering."
        for key in ADVICE:
            if key.replace(" ", "_").lower() in disease_raw.lower():
                advice = ADVICE[key]
                break

    result = {
        "crop": crop,
        "status": status,
        "disease": disease,
        "severity": severity,
        "confidence": confidence,
        "advice": advice,
    }

    return jsonify(result)

# =============================
# API: Generate PDF
# =============================
@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    data = request.json

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Crop Health Diagnostic Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Crop: {data['crop']}", ln=True)
    pdf.cell(0, 10, f"Status: {data['status']}", ln=True)
    pdf.cell(0, 10, f"Disease: {data['disease']}", ln=True)
    pdf.cell(0, 10, f"Severity: {data['severity']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {round(data['confidence'] * 100, 2)}%", ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 10, f"Advice:\n{data['advice']}")

    # Save PDF to backend directory
    filename = os.path.join(SCRIPT_DIR, "crop_report.pdf")
    pdf.output(filename)

    return send_file(filename, as_attachment=True, download_name="crop_report.pdf")

# =============================
# Run app
# =============================
if __name__ == "__main__":
    app.run(debug=True)
