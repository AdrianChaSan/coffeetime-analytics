import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CoffeeTime Café", page_icon="☕")
st.title("CoffeeTime Café")
st.subheader("Predicción de retorno de clientes")
st.write("Estima si un cliente regresará durante la próxima semana.")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input(
            "Edad",
            min_value=18,
            max_value=100,
            value=30,
            help="Rango observado en los datos de entrenamiento: 18 a 64 años.",
        )
        visit_frequency = st.number_input(
            "Frecuencia de visitas",
            min_value=0.0,
            value=3.0,
            step=0.5,
            help="Frecuencia semanal promedio (rango en entrenamiento: 1.0 a 7.0).",
        )
        satisfaction_score = st.number_input(
            "Calificación de satisfacción (1-5)",
            min_value=1,
            max_value=5,
            value=4,
            help="Escala de 1 (muy baja) a 5 (muy alta).",
        )
        total_monthly_spent = st.number_input(
            "Gasto mensual total (USD)",
            min_value=0.0,
            value=200.0,
            step=10.0,
            help="Rango observado en entrenamiento: 53.20 a 855.04 USD.",
        )
        promo_used_label = st.selectbox("¿Usó promoción?", ["No", "Sí"])
        avg_wait_time_min = st.number_input(
            "Tiempo de espera promedio (min)",
            min_value=0.0,
            value=6.0,
            step=0.5,
            help="Rango observado en entrenamiento: 1.0 a 15.6 minutos.",
        )
    with c2:
        days_since_last_visit = st.number_input(
            "Días desde última visita",
            min_value=0,
            value=7,
            help="Rango observado en entrenamiento: 0 a 30 días.",
        )
        coffee_type_preferred = st.selectbox(
            "Café preferido",
            ["Espresso", "Cappuccino", "Latte", "Cold Brew"],
        )
        time_of_visit = st.selectbox(
            "Horario de visita",
            ["Manana", "Tarde", "Noche"],
            format_func=lambda x: "Mañana" if x == "Manana" else x,
        )
        store_location = st.selectbox(
            "Ubicación del local",
            ["Mall", "Residencial", "Centro", "Aeropuerto"],
        )
        membership_tier = st.selectbox(
            "Nivel de membresía",
            ["Regular", "Silver", "Gold", "Platinum"],
        )

    submitted = st.form_submit_button("Predecir retorno")

if submitted:
    promo_used = 1 if promo_used_label == "Sí" else 0
    cliente = pd.DataFrame(
        [
            {
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
        ]
    )

    pred = int(model.predict(cliente)[0])
    prob = float(model.predict_proba(cliente)[0][1])

    st.divider()
    st.metric("Probabilidad estimada de retorno", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    if pred == 1:
        st.success("El modelo estima que el cliente REGRESARÁ la próxima semana.")
    else:
        st.warning("El modelo estima que el cliente NO REGRESARÁ la próxima semana.")
    st.caption("Es una estimación del modelo, no una garantía.")