"""
app.py — Flask server for DQN trading agent inference.

Run:
    pip install flask torch
    python app.py

The model (dqn_model.pth) must be in the same directory.
"""

from flask import Flask, request, jsonify, send_from_directory
from dqn_agent import DQNAgent
import numpy as np
import os

app = Flask(__name__, static_folder=".")

MODEL_PATH = "dqn_model.pth"

# Load agent once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. Run train.py first."
    )

agent = DQNAgent.load(MODEL_PATH)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "close":   float,
        "sma":     float,
        "ema":     float,
        "rsi":     float,
        "balance": float,
        "shares":  int
    }

    Returns:
    {
        "action": 0|1|2,
        "label":  "HOLD"|"BUY"|"SELL",
        "q_values": [q0, q1, q2]
    }
    """
    data = request.get_json()

    required = ["close", "sma", "ema", "rsi", "balance", "shares"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        state = np.array([
            float(data["close"]),
            float(data["sma"]),
            float(data["ema"]),
            float(data["rsi"]),
            float(data["balance"]),
            float(data["shares"]),
        ], dtype=np.float32)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    action, label, q_values = agent.predict(state)

    return jsonify({
        "action": action,
        "label": label,
        "q_values": [round(q, 4) for q in q_values],
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_path": MODEL_PATH,
        "epsilon": round(agent.epsilon, 4),
        "state_size": agent.state_size,
        "action_size": agent.action_size,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)