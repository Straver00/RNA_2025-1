import pickle
import pandas as pd  # type: ignore
from flask import Flask, render_template, request  # type: ignore

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """Página principal del score de riesgo"""

    if request.method == "POST":
        user_values = {}

        # Captura de variables del formulario
        user_values["edad"] = float(request.form["edad"])
        user_values["ingresos"] = float(request.form["ingresos"])
        user_values["deudas"] = float(request.form["deudas"])

        df = pd.DataFrame.from_dict(user_values, orient="index").T

        # Cargamos el modelo
        #with open("modelo/red_entrenada.pkl", "rb") as file:
            #loaded_model = pickle.load(file)

        # Calculamos el score
        #score = loaded_model.predict_proba(df)[0][1]
        score = 75.0  # Simulación de score para el ejemplo

    else:
        score = None
    
    return render_template("index.html", score=score)
if __name__ == "__main__":
    app.run(debug=True)
