from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

REQUIRED_FEATURES = [
    "age",
    "visit_frequency",
    "coffee_type_preferred",
    "time_of_visit",
    "satisfaction_score",
    "total_monthly_spent",
    "store_location",
    "membership_tier",
    "promo_used",
    "avg_wait_time_min",
    "days_since_last_visit",
]

COFFEE_OPTIONS = ["Espresso", "Cappuccino", "Latte", "Cold Brew"]
TIME_OPTIONS = ["Manana", "Tarde", "Noche"]
LOCATION_OPTIONS = ["Mall", "Residencial", "Centro", "Aeropuerto"]
MEMBERSHIP_OPTIONS = ["Regular", "Silver", "Gold", "Platinum"]

PRESETS = {
    "vip": {
        "label": "🌟 Cliente VIP frecuente",
        "age": 34,
        "visit_frequency": 6.0,
        "satisfaction_score": 5,
        "total_monthly_spent": 580.0,
        "promo_used_label": "Sí",
        "avg_wait_time_min": 3.5,
        "days_since_last_visit": 2,
        "coffee_type_preferred": "Cold Brew",
        "time_of_visit": "Manana",
        "store_location": "Aeropuerto",
        "membership_tier": "Platinum",
    },
    "riesgo": {
        "label": "⚠️ Cliente en riesgo",
        "age": 52,
        "visit_frequency": 1.5,
        "satisfaction_score": 2,
        "total_monthly_spent": 95.0,
        "promo_used_label": "No",
        "avg_wait_time_min": 13.5,
        "days_since_last_visit": 22,
        "coffee_type_preferred": "Espresso",
        "time_of_visit": "Noche",
        "store_location": "Mall",
        "membership_tier": "Regular",
    },
    "promedio": {
        "label": "☕ Cliente promedio",
        "age": 42,
        "visit_frequency": 3.5,
        "satisfaction_score": 3,
        "total_monthly_spent": 298.0,
        "promo_used_label": "No",
        "avg_wait_time_min": 7.0,
        "days_since_last_visit": 8,
        "coffee_type_preferred": "Latte",
        "time_of_visit": "Tarde",
        "store_location": "Centro",
        "membership_tier": "Silver",
    },
}

st.set_page_config(
    page_title="CoffeeTime Café | Analítica Predictiva",
    page_icon="☕",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    clf_model = joblib.load(BASE_DIR / "model.joblib")
    reg_model = joblib.load(BASE_DIR / "reg_model.joblib")
    with open(BASE_DIR / "model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return clf_model, reg_model, metadata


def classify_risk(prob_return: float) -> tuple[str, str]:
    if prob_return >= 0.65:
        return "Bajo", "Cliente con alta intención de retorno"
    if prob_return >= 0.40:
        return "Moderado", "Retorno incierto; sensible a incentivos"
    return "Alto", "Alta probabilidad de abandono semanal"


def generate_recommendations(row: dict, prob_return: float, pred_ticket: float, metadata: dict) -> list[str]:
    recs: list[str] = []
    med_days = metadata["dataset"]["numeric_summary"]["days_since_last_visit"]["median"]
    med_freq = metadata["dataset"]["numeric_summary"]["visit_frequency"]["median"]
    med_spent = metadata["dataset"]["numeric_summary"]["total_monthly_spent"]["median"]

    if row["days_since_last_visit"] > med_days:
        recs.append(
            f"**Campaña de reactivación inmediata:** lleva {row['days_since_last_visit']} días sin visitar "
            f"(mediana: {med_days:.0f} días). Enviar cupón personalizado válido por 7 días."
        )
    if row["satisfaction_score"] <= 2 or row["avg_wait_time_min"] >= 10.0:
        recs.append(
            f"**Mejora de experiencia en tienda ({row['store_location']}):** satisfacción de "
            f"{row['satisfaction_score']}/5 y espera de {row['avg_wait_time_min']:.1f} min. "
            "Priorizar atención ágil o compensación en la próxima compra."
        )
    if row["visit_frequency"] < med_freq and prob_return < 0.65:
        recs.append(
            "**Incentivo de frecuencia semanal:** ofrecer tarjeta de sellos o beneficio por "
            "visitas consecutivas en 7 días para consolidar el hábito."
        )
    if row["total_monthly_spent"] >= med_spent or pred_ticket >= metadata["dataset"]["avg_spent_mean"]:
        recs.append(
            f"**Estrategia de upselling dirigida:** su gasto mensual (${row['total_monthly_spent']:.0f}) y "
            f"ticket estimado (${pred_ticket:.2f}) son competitivos; sugerir combos de repostería o "
            f"bebidas premium junto a su {row['coffee_type_preferred']}."
        )
    if not recs:
        recs.append(
            "**Fidelización preventiva:** mantener comunicación con beneficios exclusivos para su "
            f"nivel {row['membership_tier']} en horario {format_time(row['time_of_visit'])}."
        )
    return recs


def primary_action_short(row: pd.Series, prob_return: float) -> str:
    if row["satisfaction_score"] <= 2 or row["avg_wait_time_min"] >= 10.0:
        return "Revisar experiencia / espera"
    if prob_return < 0.40:
        return "Cupón de reactivación"
    if prob_return < 0.65:
        return "Incentivo de frecuencia"
    return "Upselling / programa VIP"


def format_time(value: str) -> str:
    return "Mañana" if value == "Manana" else value


def init_form_state() -> None:
    defaults = PRESETS["promedio"]
    for field, val in defaults.items():
        if field == "label":
            continue
        key = f"input_{field}"
        if key not in st.session_state:
            st.session_state[key] = val
    if "has_prediction" not in st.session_state:
        st.session_state["has_prediction"] = False


def apply_preset(preset_key: str) -> None:
    preset = PRESETS[preset_key]
    for field, val in preset.items():
        if field == "label":
            continue
        st.session_state[f"input_{field}"] = val
    st.session_state["has_prediction"] = True


def build_sample_batch_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "DEMO_001",
                "age": 34,
                "visit_frequency": 6.0,
                "coffee_type_preferred": "Cold Brew",
                "time_of_visit": "Manana",
                "satisfaction_score": 5,
                "total_monthly_spent": 580.0,
                "store_location": "Aeropuerto",
                "membership_tier": "Platinum",
                "promo_used": 1,
                "avg_wait_time_min": 3.5,
                "days_since_last_visit": 2,
            },
            {
                "customer_id": "DEMO_002",
                "age": 52,
                "visit_frequency": 1.5,
                "coffee_type_preferred": "Espresso",
                "time_of_visit": "Noche",
                "satisfaction_score": 2,
                "total_monthly_spent": 95.0,
                "store_location": "Mall",
                "membership_tier": "Regular",
                "promo_used": 0,
                "avg_wait_time_min": 13.5,
                "days_since_last_visit": 22,
            },
            {
                "customer_id": "DEMO_003",
                "age": 41,
                "visit_frequency": 3.8,
                "coffee_type_preferred": "Latte",
                "time_of_visit": "Tarde",
                "satisfaction_score": 4,
                "total_monthly_spent": 340.0,
                "store_location": "Centro",
                "membership_tier": "Gold",
                "promo_used": 1,
                "avg_wait_time_min": 6.0,
                "days_since_last_visit": 5,
            },
            {
                "customer_id": "DEMO_004",
                "age": 26,
                "visit_frequency": 2.4,
                "coffee_type_preferred": "Cappuccino",
                "time_of_visit": "Manana",
                "satisfaction_score": 3,
                "total_monthly_spent": 190.0,
                "store_location": "Residencial",
                "membership_tier": "Silver",
                "promo_used": 0,
                "avg_wait_time_min": 8.5,
                "days_since_last_visit": 12,
            },
            {
                "customer_id": "DEMO_005",
                "age": 48,
                "visit_frequency": 5.2,
                "coffee_type_preferred": "Espresso",
                "time_of_visit": "Manana",
                "satisfaction_score": 5,
                "total_monthly_spent": 490.0,
                "store_location": "Centro",
                "membership_tier": "Gold",
                "promo_used": 0,
                "avg_wait_time_min": 4.2,
                "days_since_last_visit": 3,
            },
        ]
    )


def run_batch_predictions(
    df_input: pd.DataFrame, clf_model, reg_model, threshold: float
) -> pd.DataFrame:
    df_clean = df_input.copy()
    if df_clean["promo_used"].dtype == object or str(df_clean["promo_used"].dtype) == "str":
        df_clean["promo_used"] = (
            df_clean["promo_used"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"sí": 1, "si": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0})
            .fillna(0)
            .astype(int)
        )

    X_batch = df_clean[REQUIRED_FEATURES]
    probs = clf_model.predict_proba(X_batch)[:, 1]
    preds = (probs >= threshold).astype(int)
    tickets = reg_model.predict(X_batch)

    df_out = df_clean.copy()
    df_out["prob_retorno"] = probs.round(4)
    df_out["pred_retorno"] = ["Regresa" if p == 1 else "No regresa" for p in preds]
    df_out["riesgo_abandono"] = [classify_risk(float(p))[0] for p in probs]
    df_out["ticket_estimado_usd"] = tickets.round(2)
    df_out["accion_sugerida"] = [
        primary_action_short(df_clean.iloc[i], float(probs[i])) for i in range(len(df_clean))
    ]
    return df_out


clf_model, reg_model, metadata = load_artifacts()
init_form_state()

# --- Sidebar ---
with st.sidebar:
    st.title("☕ CoffeeTime Café")
    st.caption("Data Product — Etapa de Despliegue (CRISP-DM)")
    st.divider()

    threshold = st.slider(
        "Umbral de decisión (retorno)",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help=(
            "Probabilidad mínima para clasificar que un cliente regresará. "
            "Reducir el umbral aumenta el recall; subirlo prioriza precisión."
        ),
    )

    st.divider()
    st.markdown("**Modelos desplegados**")
    st.markdown(
        f"- **Retorno semanal:** `{metadata['classification']['model_name']}` "
        f"(F1 CV: `{metadata['classification']['f1_cv_mean']:.3f}`)\n"
        f"- **Ticket promedio:** `{metadata['regression']['model_name']}` "
        f"(RMSE: `${metadata['regression']['rmse_oof']:.2f}` | R²: `{metadata['regression']['r2_oof']:.3f}`)"
    )
    st.caption(
        f"Entrenados con {metadata['dataset']['n_samples']:,} clientes "
        f"(tasa histórica de retorno: {metadata['dataset']['return_rate']:.1%})."
    )

# --- Header ---
st.title("☕ CoffeeTime Café — Analítica Predictiva de Clientes")
st.write(
    "Estima simultáneamente la **probabilidad de retorno semanal** (`return_next_week`) y el "
    "**ticket promedio esperado** (`avg_spent`), junto con recomendaciones accionables de retención."
)

tab_individual, tab_batch, tab_models = st.tabs(
    ["🔮 Predicción Individual", "📂 Predicción por Lote (CSV)", "📊 Análisis de Modelos"]
)

# ==================== TAB 1: INDIVIDUAL ====================
with tab_individual:
    st.markdown("##### Perfiles rápidos de prueba")
    p_cols = st.columns(3)
    for idx, (pkey, pdata) in enumerate(PRESETS.items()):
        with p_cols[idx]:
            st.button(
                pdata["label"],
                key=f"preset_btn_{pkey}",
                width="stretch",
                on_click=apply_preset,
                args=(pkey,),
            )

    with st.form("customer_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Edad",
                min_value=18,
                max_value=100,
                key="input_age",
                help="Rango en entrenamiento: 18 a 64 años (mediana: 42).",
            )
            visit_frequency = st.number_input(
                "Frecuencia de visitas",
                min_value=0.0,
                step=0.5,
                key="input_visit_frequency",
                help="Visitas promedio por semana (rango: 1.0 a 7.0, mediana: 3.5).",
            )
            satisfaction_score = st.number_input(
                "Calificación de satisfacción (1-5)",
                min_value=1,
                max_value=5,
                key="input_satisfaction_score",
                help="Escala de 1 (muy baja) a 5 (muy alta).",
            )
            total_monthly_spent = st.number_input(
                "Gasto mensual total (USD)",
                min_value=0.0,
                step=10.0,
                key="input_total_monthly_spent",
                help="Rango en entrenamiento: 53.20 a 855.04 USD (mediana: 297.78 USD).",
            )
            promo_used_label = st.selectbox(
                "¿Usó promoción?",
                ["No", "Sí"],
                key="input_promo_used_label",
            )
            avg_wait_time_min = st.number_input(
                "Tiempo de espera promedio (min)",
                min_value=0.0,
                step=0.5,
                key="input_avg_wait_time_min",
                help="Rango en entrenamiento: 1.0 a 15.6 min (mediana: 7.0 min).",
            )
        with c2:
            days_since_last_visit = st.number_input(
                "Días desde última visita",
                min_value=0,
                key="input_days_since_last_visit",
                help="Rango en entrenamiento: 0 a 30 días (mediana: 8 días).",
            )
            coffee_type_preferred = st.selectbox(
                "Café preferido",
                COFFEE_OPTIONS,
                key="input_coffee_type_preferred",
            )
            time_of_visit = st.selectbox(
                "Horario de visita",
                TIME_OPTIONS,
                format_func=format_time,
                key="input_time_of_visit",
            )
            store_location = st.selectbox(
                "Ubicación del local",
                LOCATION_OPTIONS,
                key="input_store_location",
            )
            membership_tier = st.selectbox(
                "Nivel de membresía",
                MEMBERSHIP_OPTIONS,
                key="input_membership_tier",
            )

        submitted = st.form_submit_button(
            "Predecir retorno y ticket promedio", type="primary", width="stretch"
        )
        if submitted:
            st.session_state["has_prediction"] = True

    if st.session_state["has_prediction"]:
        promo_used = 1 if promo_used_label == "Sí" else 0
        row_dict = {
            "age": age,
            "visit_frequency": visit_frequency,
            "coffee_type_preferred": coffee_type_preferred,
            "time_of_visit": time_of_visit,
            "satisfaction_score": satisfaction_score,
            "total_monthly_spent": total_monthly_spent,
            "store_location": store_location,
            "membership_tier": membership_tier,
            "promo_used": promo_used,
            "avg_wait_time_min": avg_wait_time_min,
            "days_since_last_visit": days_since_last_visit,
        }
        cliente_df = pd.DataFrame([row_dict])

        prob = float(clf_model.predict_proba(cliente_df)[0][1])
        pred = 1 if prob >= threshold else 0
        pred_ticket = float(reg_model.predict(cliente_df)[0])

        risk_label, risk_desc = classify_risk(prob)
        base_rate = metadata["dataset"]["return_rate"]
        hist_ticket = metadata["dataset"]["avg_spent_mean"]

        st.divider()
        st.subheader("Resultados de la evaluación")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Probabilidad de retorno",
                f"{prob:.1%}",
                delta=f"{(prob - base_rate):+.1%} vs tasa base",
            )
        with m2:
            st.metric("Riesgo de abandono", risk_label, delta=risk_desc, delta_color="off")
        with m3:
            st.metric(
                "Ticket promedio estimado",
                f"${pred_ticket:.2f} USD",
                delta=f"{(pred_ticket - hist_ticket):+.2f} USD vs promedio",
            )

        st.progress(min(max(prob, 0.0), 1.0))

        if pred == 1:
            st.success(
                f"Con umbral **{threshold:.0%}**, el modelo estima que el cliente "
                f"**REGRESARÁ** la próxima semana con un ticket cercano a **${pred_ticket:.2f} USD**."
            )
        else:
            st.warning(
                f"Con umbral **{threshold:.0%}**, el modelo estima que el cliente "
                f"**NO REGRESARÁ** la próxima semana (ticket estimado si vuelve: **${pred_ticket:.2f} USD**)."
            )

        col_rec, col_cmp = st.columns([3, 2])
        with col_rec:
            st.markdown("#### Acciones recomendadas para gerencia")
            for rec in generate_recommendations(row_dict, prob, pred_ticket, metadata):
                st.markdown(f"- {rec}")

        with col_cmp:
            st.markdown("#### Perfil vs. mediana histórica")
            ns = metadata["dataset"]["numeric_summary"]
            cmp_df = pd.DataFrame(
                [
                    {
                        "Indicador": "Frecuencia de visitas",
                        "Cliente": f"{visit_frequency:.1f}",
                        "Mediana": f"{ns['visit_frequency']['median']:.1f}",
                    },
                    {
                        "Indicador": "Gasto mensual (USD)",
                        "Cliente": f"${total_monthly_spent:.0f}",
                        "Mediana": f"${ns['total_monthly_spent']['median']:.0f}",
                    },
                    {
                        "Indicador": "Satisfacción (1-5)",
                        "Cliente": f"{satisfaction_score}",
                        "Mediana": f"{ns['satisfaction_score']['median']:.0f}",
                    },
                    {
                        "Indicador": "Tiempo de espera (min)",
                        "Cliente": f"{avg_wait_time_min:.1f}",
                        "Mediana": f"{ns['avg_wait_time_min']['median']:.1f}",
                    },
                    {
                        "Indicador": "Días desde última visita",
                        "Cliente": f"{days_since_last_visit}",
                        "Mediana": f"{ns['days_since_last_visit']['median']:.0f}",
                    },
                ]
            )
            st.dataframe(cmp_df, width="stretch", hide_index=True)

        st.caption("Estimaciones generadas por modelos estadísticos; deben usarse como apoyo a decisiones comerciales.")

# ==================== TAB 2: BATCH CSV ====================
with tab_batch:
    st.subheader("Evaluación masiva de clientes (Batch)")
    st.write(
        "Sube un archivo `.csv` con las columnas del cliente para predecir retorno, riesgo de abandono, "
        "ticket estimado y acción recomendada para todos los registros."
    )

    sample_df = build_sample_batch_df()
    b1, b2 = st.columns([1, 2])
    with b1:
        st.download_button(
            "⬇️ Descargar plantilla CSV de ejemplo",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="clientes_ejemplo_coffeetime.csv",
            mime="text/csv",
            width="stretch",
        )
    with b2:
        use_demo_batch = st.checkbox(
            "Usar los 5 clientes de ejemplo integrados para probar sin subir archivo",
            value=False,
        )

    uploaded_file = st.file_uploader("O sube tu archivo CSV", type=["csv"])

    batch_source = None
    if uploaded_file is not None:
        batch_source = pd.read_csv(uploaded_file)
    elif use_demo_batch:
        batch_source = sample_df

    if batch_source is not None:
        missing_cols = [c for c in REQUIRED_FEATURES if c not in batch_source.columns]
        if missing_cols:
            st.error(f"El CSV no contiene las columnas obligatorias: {', '.join(missing_cols)}")
        else:
            results_df = run_batch_predictions(batch_source, clf_model, reg_model, threshold)
            n_total = len(results_df)
            n_return = int((results_df["pred_retorno"] == "Regresa").sum())
            n_high_risk = int((results_df["riesgo_abandono"] == "Alto").sum())
            avg_ticket_batch = float(results_df["ticket_estimado_usd"].mean())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Clientes evaluados", f"{n_total}")
            k2.metric("Estiman regresar", f"{n_return} ({n_return / n_total:.0%})")
            k3.metric("Alto riesgo de abandono", f"{n_high_risk}")
            k4.metric("Ticket promedio estimado", f"${avg_ticket_batch:.2f} USD")

            st.dataframe(results_df, width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Descargar resultados en CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_coffeetime.csv",
                mime="text/csv",
                type="primary",
            )

# ==================== TAB 3: MODEL ANALYSIS ====================
with tab_models:
    st.subheader("Desempeño e interpretabilidad de los modelos")
    clf_meta = metadata["classification"]
    reg_meta = metadata["regression"]

    col_clf, col_reg = st.columns(2)
    with col_clf:
        st.markdown(f"#### Clasificación — `{clf_meta['model_name']}`")
        st.caption(f"Target: `{clf_meta['target']}` | Validación cruzada estratificada (k={clf_meta['cv_folds']})")
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("F1-Score CV", f"{clf_meta['f1_cv_mean']:.3f}")
        cm2.metric("Accuracy", f"{clf_meta['accuracy_oof']:.1%}")
        cm3.metric("Precision (1)", f"{clf_meta['precision_pos_oof']:.1%}")
        cm4.metric("Recall (1)", f"{clf_meta['recall_pos_oof']:.1%}")

        st.markdown("**Top variables más influyentes**")
        clf_imp_df = (
            pd.DataFrame(clf_meta["top_features"])
            .set_index("label")[["importance"]]
            .rename(columns={"importance": "Importancia"})
        )
        st.bar_chart(clf_imp_df, horizontal=True)

        st.markdown("**Matriz de confusión (Out-of-Fold)**")
        cm = clf_meta["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Real: No regresa", "Real: Regresa"],
            columns=["Pred: No regresa", "Pred: Regresa"],
        )
        st.dataframe(cm_df, width="stretch")

    with col_reg:
        st.markdown(f"#### Regresión — `{reg_meta['model_name']}`")
        st.caption(f"Target: `{reg_meta['target']}` | Validación cruzada K-Fold (k={reg_meta['cv_folds']})")
        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("RMSE", f"${reg_meta['rmse_oof']:.2f}")
        rm2.metric("MAE", f"${reg_meta['mae_oof']:.2f}")
        rm3.metric("R²", f"{reg_meta['r2_oof']:.3f}")

        st.markdown("**Top variables más influyentes**")
        reg_imp_df = (
            pd.DataFrame(reg_meta["top_features"])
            .set_index("label")[["importance"]]
            .rename(columns={"importance": "Importancia"})
        )
        st.bar_chart(reg_imp_df, horizontal=True)

        st.info(
            "**Lectura operativa (CRISP-DM):** `satisfaction_score`, `total_monthly_spent`, "
            "`visit_frequency`, `days_since_last_visit` y `avg_wait_time_min` explican más del 83% "
            "de la predicción de retorno. Ajustar el umbral en la barra lateral permite priorizar "
            "campañas de retención (mayor recall) cuando el costo de contactar a un cliente es bajo."
        )

    st.divider()
    ver = metadata["versions"]
    st.caption(
        f"Versiones de entrenamiento — scikit-learn `{ver['scikit_learn']}` | "
        f"pandas `{ver['pandas']}` | numpy `{ver['numpy']}` | joblib `{ver['joblib']}`"
    )
