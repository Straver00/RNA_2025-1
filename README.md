# RNA-01
Este repositorio corresponde a los recursos y código fuente empleados por el Equipo #2 del curso *Redes Neuronales Artificiales y Algortimos Bioinspirados*,
en el semestre 2025-1S, para los diferentes 4 trabajos del curso. Los cuáles consisten en:

## Trabajo 1: Optimización heurística

Entendiendo lo importante que es la optimización para diversos procesos modernos y tecnologías como las redes neuronales; el primer trabajo se centra en el estudio de métodos optimización clásicos como
el descenso por gradientes, y los algoritmos bioinspirados (GA, PSO, DE, ACO). Para esto se estudia tanto optimización numérica como combinatoria, estudiando casos teóricos y
reales aplicados.

  - Lenguaje empleado: R
  - Archivo con el código fuente: RNA-01.Rmd
  - Archivo web del blog: RNA-01.html
  - Los demás archivos son los .csv con los datos del problema del vendedor viajero y los .gif de las animaciones.

## Trabajo 2: Aplicaciones de redes neuronales a datos tabulares

Uso práctico de redes neuronales con datos tabulares para la estimación de scorecards de crédito. Implementación práctica por medio de una app web con el modelo compilado realizando inferencia.

  - webpage: Carpeta con proyecto de la página web
  - bestModel.keras: Mejor modelo hallado guardado en formato .keras
  - datos.csv: Datos procesados utilizados para entrenar y evaluar el modelo
  - scalerMinMax.pkl: Escalador aplicable a los datos tabulares numéricos

## Trabajo 3: Aplicaciones en sistemas de recomendación e imágenes

## Sistema Inteligente de Transporte

## 📖 Descripción
Este proyecto implementa un _Sistema Inteligente de Transporte_ basado en aprendizaje automático y una interfaz web interactiva. Proporciona tres funcionalidades principales:

1. **Clasificador de imágenes**: Procesa imágenes de rutas o vehículos para determinar su categoría de riesgo o estado.
2. **Predicción de demanda**: Utiliza una red neuronal en PyTorch para estimar la demanda de transporte en función de factores como día de la semana, hora y capacidad.
3. **Recomendación de destinos**: Emplea un modelo de recomendación en TensorFlow para sugerir los destinos más adecuados según las preferencias y perfil del usuario.

Cada módulo se integra en una aplicación web creada con Flask, ofreciendo un flujo sencillo para cargar datos, obtener predicciones y visualizar resultados.

---

## 🚀 Características Principales

- **Flask Web App**:
  - Rutas:
    - `/` → Página principal.
    - `/classify` → Formulario para subir y clasificar imágenes.
    - `/demand` → Formulario para ingresar parámetros y predecir demanda.
    - `/recommend` → Formulario para preferencias de viaje y mostrar recomendaciones.
  - Plantillas HTML con Jinja2 (`templates/`).
  - Archivos estáticos (CSS, uploads) en `static/`.

- **Modelos de Machine Learning**:
  - **Clasificación de Imágenes**:
    - Modelo CNN pre-entrenado en TensorFlow (`models/CNN_final.keras`).
    - Función de preprocesamiento de imágenes en `inference_model.py`.
  - **Predicción de Demanda**:
    - Arquitectura de red neuronal (23→5→10→20→1) en PyTorch (`models/modelo_dem_pytorch.pth`).
    - Normalización de datos con `preprocessor.pkl`.
  - **Recomendación de Destinos**:
    - Modelo secuencial en TensorFlow (`models/travel_recommendation_model.keras`).
    - Encoders y escaladores: `scaler.joblib`, `gender_encoder.joblib`, `mlb.joblib`.
    - Datos de destinos en `models/Expanded_Destinations.csv`.

---

## 📁 Estructura del Proyecto
```
├── main.py                   # Punto de entrada de la aplicación Flask
├── inference_model.py        # Preprocesamiento de imágenes
├── requirements.txt          # Dependencias del proyecto
├── templates/                # Plantillas HTML (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── classify.html
│   ├── demand.html
│   └── recommend.html
├── static/                   # Archivos estáticos (CSS, imagenes subidas)
│   ├── styles.css
│   └── uploads/
├── models/                   # Modelos y artefactos de ML
│   ├── CNN_final.keras
│   ├── best_model.keras
│   ├── modelo_dem_pytorch.pth
│   ├── travel_recommendation_model.keras
│   ├── preprocessor.pkl
│   ├── scaler.joblib
│   ├── gender_encoder.joblib
│   ├── mlb.joblib
│   └── Expanded_Destinations.csv
└── .gitignore
```

---

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio**:
   ```bash
   git clone <URL-del-repositorio>
   cd <nombre-del-repo>
   ```

2. **Crear un entorno virtual**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   .\venv\Scripts\activate    # Windows
   ```

3. **Instalar dependencias**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


5. **Variables de entorno (opcional)**:
   - `UPLOAD_FOLDER`: Ruta para guardar imágenes; por defecto `static/uploads/`.
   - `MODELS_DIR`: Directorio donde se encuentran los modelos; por defecto `models/`.

---

## ▶️ Uso

Iniciar la aplicación:
```bash
python main.py
```

Visitar en el navegador: `http://localhost:5000`

1. **Clasificación de Imágenes**:
   - Navegar a `/classify`.
   - Subir una imagen y obtener la categoría.

2. **Predicción de Demanda**:
   - Navegar a `/demand`.
   - Rellenar los campos y enviar para ver la predicción.

3. **Recomendación de Destinos**:
   - Navegar a `/recommend`.
   - Ingresar datos de perfil y preferencias, obtener lista de destinos.

---

## 📊 Detalles Técnicos de los Modelos

- **CNN de Clasificación**: Entrenada para reconocer patrones en imágenes de rutas; entrada escalada a 200×200 píxeles en escala de grises.
- **Red de Demanda (PyTorch)**: Cuatro capas densas con activación ReLU, salida escalar.
- **Recomendación (TensorFlow)**: Modelo secuencial que combina características numéricas, codificaciones de género y preferencias, junto con atributos de destinos.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 📫 Contacto

Para dudas o sugerencias, contacta al equipo de desarrollo:
- Email: stjuliod@unal.edu.co
- GitHub: https://github.com/stjuliod09/RNA_Trabajo-3

## Trabajo 4: Aplicaciones de grandes modelos de lenguaje

Este min-pryecto trata básicamente de la creación de un agente con arquitectura rápida y sencilla (sin framework específico de agentes IA), el cual tiene la capacidad de asistir con tareas de generación de historias. Para esto emplea la API de Gemini, la cual permite con sus múltiples modelos generar tanto texto como imagen.

La estructura del proyecto es:

- `story_generator.py` : Archivo con las funciones para generar la historia, usando las SDK de Google para Gemini, y haciendo controles de filtros, flujos y captación de errores.
- `app.py`: Script del aplicativo web con framework de Flask.
