import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import joblib
import pandas as pd
from kan import KAN
from flask import Flask, request, jsonify

def predict(data):
    scaler = joblib.load('scaler.pkl')

    interest_rate = data["interest_rate"]
    if interest_rate <= scaler.data_min_[0]:
        data["interest_rate"] = scaler.data_min_[0]

    if interest_rate > scaler.data_max_[0]:
        return jsonify({"result": "N/A", "prob": 0, "message": f"interest rate value: {str(interest_rate)} is out of range {str(scaler.data_max_[0])}"})

    debt_to_income_ratio = data["debt_to_income_ratio"]
    if debt_to_income_ratio <= scaler.data_min_[1]:
        data["debt_to_income_ratio"] = scaler.data_min_[1]

    if debt_to_income_ratio > scaler.data_max_[1]:
        return jsonify({"result": "N/A", "prob": 0, "message": f"debt to income ratio value: {str(debt_to_income_ratio)} is out of range {str(scaler.data_max_[1])}"})

    loan_to_value_ratio = data["loan_to_value_ratio"]
    if loan_to_value_ratio <= scaler.data_min_[2]:
        data["loan_to_value_ratio"] = scaler.data_min_[2]

    if loan_to_value_ratio > scaler.data_max_[2]:
        return jsonify({"result": "N/A", "prob": 0, "message": f"loan to value ratio value: {str(loan_to_value_ratio)} is out of range {str(scaler.data_max_[2])}"})

    df = pd.DataFrame([data])
    X = df[list(df.columns.drop("action_taken"))[0:14]]

    x = X.values
    x_scaled = scaler.transform(x)
    X = pd.DataFrame(x_scaled)

    device = torch.device("cpu")
    model = KAN.loadckpt(device, './kan-model')
    val_input = torch.tensor(X.to_numpy(), dtype=torch.float32, device=device)

    model_result = model.forward(val_input).detach()

    class_prob = torch.softmax(model_result, dim=1)
    accepted_prob = int(class_prob.cpu().numpy()[0][1] * 100.0)

    if accepted_prob >= 80:
        applicationStatus = "accepted"
    else:
        applicationStatus = "declined"

    json = jsonify({"result": applicationStatus, "prob": accepted_prob, "message": ""})

    return json

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            data = request.get_json()
            result = predict(data)
            return result
        except Exception as e:
            return jsonify({"result": "N/A", "prob": 0, "message": str(e)})

    return "OK"

if __name__ == "__main__":
    app.run(debug=True)