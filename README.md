# CoffeeTime Café — Predicción de Retorno de Clientes

Aplicación web construida con **Streamlit** para predecir si un cliente de **CoffeeTime Café** regresará durante la próxima semana (`return_next_week`), utilizando un modelo de clasificación preentrenado con **scikit-learn**.

## Descripción del proyecto

- **Problema:** Clasificación binaria (`1` = el cliente regresa la próxima semana, `0` = no regresa).
- **Dataset:** 1,865 registros limpios de comportamiento de clientes (`coffee_sales_clean.csv`, 36.5% clase positiva / 63.5% clase negativa).
- **Modelo desplegado:** `Pipeline` (`ColumnTransformer` con `OneHotEncoder` + `GradientBoostingClassifier(n_estimators=100, random_state=42)`), guardado en `model.joblib`.
- **Evaluación (validación cruzada estratificada, k=5):**
  - F1-Score promedio: `0.5239`
  - Accuracy: `0.70`
  - Clase `1` (Regresa): Precision `0.63`, Recall `0.45`, F1 `0.52`
  - Clase `0` (No regresa): Precision `0.73`, Recall `0.85`, F1 `0.78`

> **Nota sobre interpretación:** Dado que el conjunto de datos presenta desbalance de clases y el recall de la clase positiva es moderado (`0.45`), la aplicación muestra tanto la clase estimada como la probabilidad predicha (`predict_proba`) para apoyar la toma de decisiones de fidelización.

## Variables de entrada

- **Numéricas:** `age`, `visit_frequency`, `satisfaction_score`, `total_monthly_spent`, `promo_used`, `avg_wait_time_min`, `days_since_last_visit`.
- **Categóricas:** `coffee_type_preferred` (`Espresso`, `Cappuccino`, `Latte`, `Cold Brew`), `time_of_visit` (`Manana`, `Tarde`, `Noche`), `store_location` (`Mall`, `Residencial`, `Centro`, `Aeropuerto`), `membership_tier` (`Regular`, `Silver`, `Gold`, `Platinum`).

## Estructura del repositorio

```text
.
├── app.py              # Aplicación web en Streamlit
├── model.joblib        # Pipeline entrenado (preprocesamiento + GradientBoostingClassifier)
├── CoffeeTime.ipynb    # Notebook de análisis, comparación de modelos y exportación
├── requirements.txt    # Dependencias fijadas para despliegue
├── .gitignore
└── README.md
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Conectar este repositorio público en [https://share.streamlit.io/](https://share.streamlit.io/).
2. Configurar:
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Hacer clic en **Deploy** para generar la URL pública `.streamlit.app`.
