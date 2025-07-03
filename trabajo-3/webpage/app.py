from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ─── PyTorch ────────────────────────────────────────────────────────
import torch
import torch.nn as nn

# Función de pre-proceso para el clasificador de imágenes
from inference_model import preprocess_image_for_model

# -------------------------------------------------------------------
# Configuración general
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1.  Clasificador de conducción distraída (TensorFlow)
# -------------------------------------------------------------------
CLASSES = [
    "Conducción Segura",
    "Hablando por Teléfono",
    "Texteando por Teléfono",
    "Imprudencia al Volante",
    "Otro riesgo",
]
classification_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "CNN_final.keras")
)

# -------------------------------------------------------------------
# 2.  Predictor de demanda (PyTorch)
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))


class PrediccionDemanda(nn.Module):
    """Red neuronal: 23 → 5 → 10 → 20 → 1  (ajusta si cambian los pesos)."""

    def __init__(self, n_input_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Cargamos los pesos (state_dict)
state_dict = torch.load(
    os.path.join(MODELS_DIR, "modelo_dem_pytorch.pth"), map_location=device
)

# Averiguamos cuántas columnas produce el preprocesador
n_features = len(preprocessor.get_feature_names_out())

demand_model = PrediccionDemanda(n_features).to(device)
demand_model.load_state_dict(state_dict)
demand_model.eval()  # modo inferencia


# -------------------------------------------------------------------
# Función de inferencia para demanda
# -------------------------------------------------------------------
def predict_demand(raw_form: dict) -> float:
    """Convierte el formulario en DataFrame → tensor → predicción escalar."""
    df = pd.DataFrame([raw_form])  # una sola fila
    X = preprocessor.transform(df)  # ndarray (1, n_features)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = demand_model(X_tensor).cpu().numpy().squeeze()

    # Si el target estaba en alguna escala diferente, des-normalízalo aquí
    return float(y_pred)


# -------------------------------------------------------------------
# Rutas Flask
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------- Clasificador de imágenes ----------
@app.route("/classify", methods=["GET", "POST"])
def classify():
    label = filename = error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            error = "No se seleccionó imagen."
        else:
            original_name = secure_filename(file.filename)
            base, _ = os.path.splitext(original_name)
            filename = f"{base}.png"
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            try:
                img = Image.open(path).convert("RGB")
                img.save(path, format="PNG")  # normaliza extensión
                img_arr = preprocess_image_for_model(path)
                preds = classification_model.predict(img_arr)
                label = CLASSES[np.argmax(preds)]
            except (UnidentifiedImageError, Exception):
                os.remove(path)
                filename = None
                error = "Error al procesar la imagen."

    return render_template("classify.html", label=label, filename=filename, error=error)


# ---------- Predicción de demanda ----------
@app.route("/demand", methods=["GET", "POST"])
def demand():
    prediction = error = None

    if request.method == "POST":
        try:
            numeric_fields = {"max_capacity", "day_of_week", "month", "hour"}
            form_data = {k: v for k, v in request.form.items()}
            for f in numeric_fields:
                form_data[f] = int(form_data[f])  # o float() si tu modelo usa float

            prediction = predict_demand(form_data)

        except Exception as err:
            error = f"No se pudo procesar la solicitud: {err}"

    return render_template("demand.html", prediction=prediction, error=error)


# -------------------------------------------------------------------
# Arranque local
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
