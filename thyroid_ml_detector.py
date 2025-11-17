import json
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

########################################################################
# 1. BUILT-IN THYROID DATASET (NO DOWNLOAD REQUIRED)
########################################################################

def load_builtin_dataset():
    # Synthetic dataset (TSH, T3, TT4)
    data = pd.DataFrame({
        "TSH": np.random.uniform(0.1, 10, 500),
        "T3": np.random.uniform(0.5, 6, 500),
        "TT4": np.random.uniform(20, 200, 500)
    })

    labels = []
    for i in range(500):
        tsh, t3, tt4 = data.loc[i, ["TSH", "T3", "TT4"]]

        if tsh > 4.5 and tt4 < 60:
            labels.append(2)  # Hypothyroid
        elif tsh < 0.5 and tt4 > 140:
            labels.append(1)  # Hyperthyroid
        else:
            labels.append(0)  # Normal

    data["label"] = labels
    return data


########################################################################
# 2. TRAINING FUNCTION
########################################################################

def train_model():
    print("Loading built-in thyroid dataset...")
    df = load_builtin_dataset()

    X = df[["TSH", "T3", "TT4"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("Training model...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    pipe.fit(X_train, y_train)

    print("\nModel Evaluation:")
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    dump(pipe, "thyroid_model.pkl")
    print("\nModel saved as thyroid_model.pkl\n")


########################################################################
# 3. CLI PREDICTION
########################################################################

def predict_cli():
    try:
        data = input("Enter values like this: {\"TSH\": 2.3, \"T3\": 1.1, \"TT4\": 90}\n")
        data = json.loads(data)

        model = load("thyroid_model.pkl")

        X = np.array([[data["TSH"], data["T3"], data["TT4"]]])
        pred = model.predict(X)[0]

        labels = {0: "Normal", 1: "Hyperthyroid", 2: "Hypothyroid"}

        print("\nPrediction:", pred, "-", labels[pred])

    except Exception as e:
        print("Error:", e)


########################################################################
# 4. FLASK API
########################################################################

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        model = load("thyroid_model.pkl")
        X = np.array([[data["TSH"], data["T3"], data["TT4"]]])
        pred = model.predict(X)[0]

        labels = {0: "Normal", 1: "Hyperthyroid", 2: "Hypothyroid"}

        return jsonify({
            "prediction": int(pred),
            "label": labels[pred]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


########################################################################
# 5. MAIN
########################################################################

def main():
    if len(sys.argv) < 2:
        print("Usage: python thyroid_ml_detector.py [train|predict-cli|serve]")
        return

    cmd = sys.argv[1].lower()

    if cmd == "train":
        train_model()

    elif cmd == "predict-cli":
        predict_cli()

    elif cmd == "serve":
        print("Starting server on port 5000...")
        app.run(port=5000)

    else:
        print("Invalid command. Use: train | predict-cli | serve")


if __name__ == "__main__":
    main()
