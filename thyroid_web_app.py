# thyroid_web_app.py
# Single-file Flask web app for thyroid prediction (uses built-in demo model if none saved)
# Usage:
#   python thyroid_web_app.py
#
# Open http://127.0.0.1:5000/ in your browser.

import os
import json
from flask import Flask, request, render_template_string, redirect, url_for, flash
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_PATH = "thyroid_model.pkl"

LABELS = {0: "Normal", 1: "Hyperthyroid", 2: "Hypothyroid"}

# ---------- helper: create a built-in synthetic dataset and train ----------
def create_builtin_dataset(n=500, random_state=42):
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        "TSH": rng.uniform(0.1, 10.0, size=n),
        "T3": rng.uniform(0.5, 6.0, size=n),
        "TT4": rng.uniform(20.0, 200.0, size=n),
    })
    labels = []
    for _, row in df.iterrows():
        tsh = row["TSH"]
        t3 = row["T3"]
        tt4 = row["TT4"]
        if tsh > 4.5 and tt4 < 60:
            labels.append(2)  # Hypothyroid
        elif tsh < 0.5 and tt4 > 140:
            labels.append(1)  # Hyperthyroid
        else:
            labels.append(0)  # Normal
    df["label"] = labels
    return df

def train_and_save_demo_model(path=MODEL_PATH):
    df = create_builtin_dataset(n=500, random_state=1)
    X = df[["TSH", "T3", "TT4"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipe.fit(X_train, y_train)
    dump(pipe, path)
    return pipe

def load_or_train_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            model = load(path)
            return model, False
        except Exception as e:
            # if loading fails, re-train
            print("Failed to load model, retraining. Error:", e)
    model = train_and_save_demo_model(path)
    return model, True

# ---------- Flask app and templates ----------
app = Flask(__name__)
app.secret_key = "change-me-for-production"  # for flash messages

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Thyroid Detector</title>
  <style>
    :root { --accent: #0b5fff; --muted:#6b7280; font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    body{background:#f6f8fa;margin:0;padding:0;display:flex;min-height:100vh;align-items:center;justify-content:center}
    .card{width:min(760px,94%);background:#fff;border-radius:12px;padding:1.25rem;box-shadow:0 8px 30px rgba(2,6,23,0.08)}
    h1{margin:0 0 0.25rem 0;font-size:1.25rem}
    p.lead{margin:0 0 1rem 0;color:var(--muted)}
    form{display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem}
    label{font-size:0.85rem;color:#111}
    input[type="number"]{padding:0.6rem;border:1px solid #e6eaee;border-radius:8px;font-size:0.95rem;width:100%}
    .full{grid-column:1/-1}
    button{background:var(--accent);color:white;padding:0.6rem 0.9rem;border:none;border-radius:8px;cursor:pointer}
    .result{margin-top:1rem;padding:0.8rem;border-radius:8px;background:#f3f7ff;border:1px solid #e6efff}
    .muted{color:var(--muted);font-size:0.9rem}
    footer{margin-top:0.75rem;text-align:center;color:var(--muted);font-size:0.85rem}
    @media (max-width:600px){ form{grid-template-columns:1fr} }
  </style>
</head>
<body>
  <div class="card">
    <h1>Thyroid Detector (Demo)</h1>
    <p class="lead">Enter TSH, T3 and TT4 values. This is a demo model — not a medical diagnosis tool.</p>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="result"><strong>{{ messages[0] }}</strong></div>
      {% endif %}
    {% endwith %}

    <form method="post" action="{{ url_for('predict') }}">
      <div>
        <label for="tsh">TSH</label>
        <input id="tsh" name="TSH" step="any" type="number" required value="{{ request.form.get('TSH', '') }}">
      </div>
      <div>
        <label for="t3">T3</label>
        <input id="t3" name="T3" step="any" type="number" required value="{{ request.form.get('T3', '') }}">
      </div>
      <div>
        <label for="tt4">TT4</label>
        <input id="tt4" name="TT4" step="any" type="number" required value="{{ request.form.get('TT4', '') }}">
      </div>

      <div class="full" style="display:flex;gap:0.5rem;align-items:center;justify-content:flex-start;margin-top:0.5rem">
        <button type="submit">Get Prediction</button>
        <a href="{{ url_for('serve_info') }}" style="align-self:center;text-decoration:none;color:var(--muted);font-size:0.9rem">Model info</a>
      </div>
    </form>

    {% if prediction is defined %}
      <div class="result">
        <div><strong>Prediction:</strong> {{ prediction }} ({{ label }})</div>
        <div style="margin-top:0.5rem"><strong>Probabilities:</strong></div>
        <pre style="background:#fff;padding:0.5rem;border-radius:6px;margin-top:0.25rem">{{ probs }}</pre>
      </div>
    {% endif %}

    <footer>
      Demo — not medical advice. Model auto-trains if none found.
    </footer>
  </div>
</body>
</html>
"""

INFO_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Model Info</title>
<style>body{font-family:system-ui;padding:1rem;background:#f6f8fa} .card{background:white;padding:1rem;border-radius:8px;max-width:900px;margin:auto}</style>
</head><body>
<div class="card">
  <h2>Model info</h2>
  <p class="muted">This demo model uses features <code>TSH</code>, <code>T3</code>, and <code>TT4</code>. Labels: <strong>0=Normal,1=Hyperthyroid,2=Hypothyroid</strong>.</p>
  <p>Model file: <code>{{ model_path }}</code></p>
  <p>When the app starts it will attempt to load the file. If missing, the app trains a small built-in demo model and saves it.</p>
  <p><a href="{{ url_for('index') }}">← Back</a></p>
</div>
</body></html>
"""

# Load or train model at startup (so first request is fast)
model, trained_now = load_or_train_model(MODEL_PATH)
if trained_now:
    print("Trained demo model and saved to", MODEL_PATH)
else:
    print("Loaded model from", MODEL_PATH)


# ---------- Flask routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        tsh = request.form.get("TSH", "").strip()
        t3 = request.form.get("T3", "").strip()
        tt4 = request.form.get("TT4", "").strip()
        # Basic validation
        if not tsh or not t3 or not tt4:
            flash("Please provide TSH, T3 and TT4 values.")
            return redirect(url_for("index"))
        try:
            tsh_v = float(tsh)
            t3_v = float(t3)
            tt4_v = float(tt4)
        except ValueError:
            flash("Invalid numeric values. Use numbers like 2.3 or 90.")
            return redirect(url_for("index"))

        # Prepare DataFrame with correct column order
        X = pd.DataFrame([[tsh_v, t3_v, tt4_v]], columns=["TSH", "T3", "TT4"])
        preds = model.predict(X)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0].tolist()

        pred = int(preds[0])
        label = LABELS.get(pred, str(pred))
        probs_text = json.dumps({ LABELS[i]: float(p) for i,p in enumerate(probs) }) if probs is not None else "N/A"

        return render_template_string(INDEX_HTML, prediction=pred, label=label, probs=probs_text)
    except Exception as e:
        flash("Server error: " + str(e))
        return redirect(url_for("index"))

@app.route("/info")
def serve_info():
    return render_template_string(INFO_HTML, model_path=MODEL_PATH)

# ---------- run ----------
import webbrowser
import threading

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run()
