# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import json

app = Flask(__name__)
CORS(app)

# Load the trained models
model_lr = joblib.load("model_lr.pkl")
model_rf = joblib.load("model_rf.pkl")
model_xgb = joblib.load("model_xgb.pkl")
kmeans = joblib.load("kmeans.pkl")

# Load metrics from file
with open("data/metrics.json", "r") as f:
    METRICS = json.load(f)

@app.route("/")
def home():
    return "Airbnb ML API: LR, RF, XGB + KMeans"

@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(METRICS)

def parse_input(data):
    return np.array([
        data["accommodates"],
        data["bathrooms"],
        data["bedrooms"],
        data["latitude"],
        data["longitude"]
    ]).reshape(1, -1)

@app.route("/predict_lr", methods=["POST"])
def predict_lr():
    try:
        data = request.get_json()
        pred = model_lr.predict(parse_input(data))[0]
        return jsonify({"predicted_price": round(float(pred), 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/predict_rf", methods=["POST"])
def predict_rf():
    try:
        data = request.get_json()
        pred = model_rf.predict(parse_input(data))[0]
        return jsonify({"predicted_price": round(float(pred), 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/predict_xgb", methods=["POST"])
def predict_xgb():
    try:
        data = request.get_json()
        pred = model_xgb.predict(parse_input(data))[0]
        return jsonify({"predicted_price": round(float(pred), 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/cluster", methods=["POST"])
def cluster():
    try:
        data = request.get_json()
        x_input = np.array([
            data["latitude"],
            data["longitude"],
            data["accommodates"]
        ]).reshape(1, -1)

        cluster_label = kmeans.predict(x_input)[0]

        # Label mapping
        cluster_labels = {
            0: "Economy Properties",
            1: "High-Class Properties"
        }

        return jsonify({
            "cluster": int(cluster_label),
            "label": cluster_labels.get(int(cluster_label), f"Group {cluster_label}")
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
