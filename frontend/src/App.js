import React, { useState } from "react";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const sendToBackend = async () => {
    if (!selectedImage) {
      alert("Please select an image first!");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("image", selectedImage);

    const res = await fetch("http://127.0.0.1:5000/api/crop-diagnosis", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  const downloadPDF = async () => {
    const res = await fetch("http://127.0.0.1:5000/api/generate-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "crop_report.pdf";
    a.click();
  };

  const statusColor = result?.status === "Healthy" ? "#2ecc71" : "#e74c3c";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #dff9fb, #c7ecee)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Segoe UI, sans-serif",
      }}
    >
      <div
        style={{
          background: "white",
          padding: 30,
          borderRadius: 16,
          width: 800,
          boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
        }}
      >
        <h1 style={{ textAlign: "center" }}>🌱 Crop Disease Detection</h1>
        <p style={{ textAlign: "center", color: "#666" }}>
          Upload a leaf image and let AI analyze it
        </p>

        <div style={{ display: "flex", gap: 30, marginTop: 30 }}>
          {/* LEFT: IMAGE */}
          <div style={{ flex: 1, textAlign: "center" }}>
            {preview ? (
              <img
                src={preview}
                alt="preview"
                style={{
                  width: "100%",
                  maxWidth: 300,
                  borderRadius: 12,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
                }}
              />
            ) : (
              <div
                style={{
                  height: 250,
                  border: "2px dashed #ccc",
                  borderRadius: 12,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#999",
                }}
              >
                No image selected
              </div>
            )}
          </div>

          {/* RIGHT: CONTROLS */}
          <div style={{ flex: 1 }}>
            <input type="file" onChange={handleImageChange} />

            <div style={{ marginTop: 20 }}>
              <button
                onClick={sendToBackend}
                style={{
                  width: "100%",
                  background: "#27ae60",
                  color: "white",
                  border: "none",
                  padding: "14px",
                  borderRadius: 10,
                  fontSize: 16,
                  cursor: "pointer",
                }}
              >
                {loading ? "Analyzing..." : "Analyze Crop"}
              </button>
            </div>
          </div>
        </div>

        {/* RESULT */}
        {result && (
          <div
            style={{
              marginTop: 30,
              padding: 25,
              borderRadius: 12,
              background: "#f7f9fa",
              border: "1px solid #ddd",
            }}
          >
            <h2>📊 Result</h2>

            <p><b>Crop:</b> {result.crop}</p>
            <p>
              <b>Status:</b>{" "}
              <span style={{ color: statusColor, fontWeight: "bold" }}>
                {result.status}
              </span>
            </p>
            <p><b>Disease:</b> {result.disease}</p>
            <p><b>Severity:</b> {result.severity}</p>
            <p><b>Confidence:</b> {(result.confidence * 100).toFixed(2)}%</p>

            <div
              style={{
                height: 10,
                width: "100%",
                background: "#ddd",
                borderRadius: 10,
                overflow: "hidden",
                marginBottom: 10,
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${(result.confidence * 100).toFixed(0)}%`,
                  background: statusColor,
                }}
              ></div>
            </div>

            <p style={{ color: statusColor, fontWeight: "bold" }}>
              ⚠️ {result.advice}
            </p>

            <button
              onClick={downloadPDF}
              style={{
                marginTop: 15,
                background: "#2980b9",
                color: "white",
                border: "none",
                padding: "12px 20px",
                borderRadius: 8,
                cursor: "pointer",
              }}
            >
              📄 Download PDF Report
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
