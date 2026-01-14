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
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# =============================
# Class names (MUST match training folders order)
# =============================
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

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

    # Parse label
    parts = label.split("___")
    crop = parts[0]
    disease_raw = parts[1]

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

    # Save PDF
    filename = "crop_report.pdf"
    pdf.output(filename)

    return send_file(filename, as_attachment=True)

# =============================
# Run app
# =============================
if __name__ == "__main__":
    app.run(debug=True)
