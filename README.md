# CoffeeTime Café — Analítica Predictiva de Clientes

Data Product desplegado con **Streamlit** para **CoffeeTime Café** (etapa de Despliegue en **CRISP-DM**). Integra dos modelos de Machine Learning preentrenados para predecir simultáneamente si un cliente regresará la próxima semana (`return_next_week`) y cuál será su ticket promedio estimado (`avg_spent`), acompañados de recomendaciones accionables de retención.

## Funcionalidades principales

1. **Predicción Individual:**
   - Perfiles de prueba en un clic (*Cliente VIP frecuente*, *Cliente en riesgo*, *Cliente promedio*) y formulario completo con las 11 variables del modelo.
   - Predicción dual: probabilidad de retorno semanal, nivel de riesgo de abandono y ticket promedio estimado (USD).
   - Umbral de clasificación configurable desde la barra lateral (`0.20` a `0.80`) para equilibrar precisión y recall según el costo de campaña.
   - Recomendaciones comerciales personalizadas y tabla comparativa contra la mediana histórica.
2. **Predicción por Lote — Batch CSV:**
   - Carga de archivos `.csv` con múltiples clientes (más plantilla descargable y dataset demo integrado de 5 clientes).
   - Resumen ejecutivo de KPIs del lote y descarga del CSV enriquecido con probabilidades, riesgo, ticket estimado y acción sugerida.
3. **Análisis de Modelos:**
   - Métricas de validación cruzada (k=5), matriz de confusión y ranking de variables más influyentes para clasificación y regresión.

## Modelos y métricas (validación cruzada k=5, N = 1,865)

| Tarea | Variable objetivo | Modelo | Métricas principales | Artefacto |
| :--- | :--- | :--- | :--- | :--- |
| **Clasificación** | `return_next_week` | `GradientBoostingClassifier` | F1 CV: `0.5235` · Accuracy: `70.2%` · Precision(1): `62.8%` · Recall(1): `44.9%` | `artifacts/model.joblib` |
| **Regresión** | `avg_spent` (USD) | `RandomForestRegressor` | RMSE: `$0.75 USD` · MAE: `$0.42 USD` · R²: `0.9652` | `artifacts/reg_model.joblib` |

> **Nota analítica:** El dataset tiene `36.5%` de casos positivos (`return_next_week = 1`). Por ello la app expone la probabilidad estimada (`predict_proba`) y un control de umbral en la barra lateral para adaptar la sensibilidad a campañas de fidelización.

## Estructura del repositorio

```text
.
├── app.py                        # Aplicación web en Streamlit (3 pestañas)
├── requirements.txt              # Dependencias fijadas para despliegue
├── README.md                     # Documentación del proyecto
├── .gitignore
├── artifacts/
│   ├── model.joblib              # Pipeline de clasificación (GradientBoostingClassifier)
│   ├── reg_model.joblib          # Pipeline de regresión (RandomForestRegressor)
│   └── model_metadata.json       # Métricas CV, importancia de variables y medianas agregadas
├── src/
│   └── train_model.py            # Script reproducible de entrenamiento y exportación
└── notebooks/
    └── CoffeeTime.ipynb          # Notebook exploratorio y comparativa de modelos
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (Opcional) Reentrenar modelos si cuentas con data/coffee_sales_clean.csv
python src/train_model.py

# Iniciar aplicación web
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Conectar este repositorio público en [https://share.streamlit.io/](https://share.streamlit.io/).
2. Configurar:
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Presionar **Deploy** para obtener la URL pública `.streamlit.app`.
