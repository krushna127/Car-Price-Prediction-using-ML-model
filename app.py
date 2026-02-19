import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

st.title("🚗 Car Price Prediction System")
st.markdown("### Predict car price based on specifications")

@st.cache_data
def load_data():
    return pd.read_csv("car_price_prediction_.csv")

df = load_data()

X = df.drop("Price", axis=1)
y = df["Price"]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

st.sidebar.header("Enter Car Details")

year = st.sidebar.number_input("Year", 2000, 2025, 2020)
engine_size = st.sidebar.number_input("Engine Size (L)", 0.5, 10.0, 2.0)
mileage = st.sidebar.number_input("Mileage (KM)", 0, 300000, 30000)

brand = st.sidebar.selectbox("Brand", sorted(df["Brand"].unique()))

filtered_models = df[df["Brand"] == brand]["Model"].unique()
model_name = st.sidebar.selectbox("Model", sorted(filtered_models))

fuel = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
transmission = st.sidebar.selectbox("Transmission", df["Transmission"].unique())
condition = st.sidebar.selectbox("Condition", df["Condition"].unique())

if st.sidebar.button("Predict Price"):

    input_data = {
        "Year": year,
        "Engine Size": engine_size,
        "Mileage": mileage,
        "Brand": brand,
        "Fuel Type": fuel,
        "Transmission": transmission,
        "Condition": condition,
        "Model": model_name
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### 💰 Estimated Car Price: ₹ {round(prediction, 2)}")

    st.markdown("---")
    st.header("📊 Estimated Price Meter")

    max_price = df["Price"].max() * 1.2

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Estimated Price"},
        number={'prefix': "₹ "},
        gauge={
            'axis': {'range': [0, max_price]},
            'bar': {'color': "#e74c3c"},
            'steps': [
                {'range': [0, max_price * 0.3], 'color': "lightgray"},
                {'range': [max_price * 0.3, max_price * 0.6], 'color': "lightyellow"},
                {'range': [max_price * 0.6, max_price], 'color': "lightgreen"}
            ],
        }
    ))

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("📈 Price Analysis")

    lower_range = prediction * 0.9
    upper_range = prediction * 1.1

    st.info(f"""
    **Estimated Market Range:**  
    ₹ {round(lower_range, 2)}  -  ₹ {round(upper_range, 2)}

    This represents possible variation in market conditions.
    """)

    st.subheader("🔎 Key Pricing Factors")

    factors = []
    car_age = 2024 - year

    if car_age <= 3:
        factors.append("✔ Newer vehicle - positively impacts resale value")
    elif car_age <= 8:
        factors.append("✔ Moderately aged vehicle - average resale value")
    else:
        factors.append("⚠ Older vehicle - higher depreciation expected")

    if mileage < 30000:
        factors.append("✔ Low mileage - increases resale value")
    elif mileage < 80000:
        factors.append("✔ Average mileage")
    else:
        factors.append("⚠ High mileage - may reduce price")

    if engine_size >= 2.5:
        factors.append("✔ Larger engine size - premium pricing")
    else:
        factors.append("✔ Standard engine size")

    if condition.lower() in ["new", "like new"]:
        factors.append("✔ Excellent condition - strong resale potential")
    else:
        factors.append("⚠ Used condition - affects negotiation value")

    for factor in factors:
        st.write(factor)

    st.markdown("---")
    st.subheader("📋 Entered Car Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Brand:** {brand}")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Year:** {year}")
        st.write(f"**Engine Size:** {engine_size} L")

    with col2:
        st.write(f"**Mileage:** {mileage} KM")
        st.write(f"**Fuel Type:** {fuel}")
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Condition:** {condition}")

else:
    st.info("Enter car details from sidebar and click Predict Price.")

st.markdown("---")
st.caption("Developed by Krushna Bhutare | Data Science Internship Project")
