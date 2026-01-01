from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Real-Time Credit Card Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expect JSON with a single transaction features or list of transactions.
    Example:
    {
        "V1": -1.359807,
        "V2": 1.191857,
        "V3": -1.358354,
        ...
        "Amount": 149.62
    }
    """
    data = request.get_json()

    # Convert input JSON to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid JSON format"}), 400

    # Scale and predict
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    preds = [0 if p == 1 else 1 for p in preds]

    return jsonify({"predictions": preds})

if __name__ == "__main__":
    app.run(debug=True)
