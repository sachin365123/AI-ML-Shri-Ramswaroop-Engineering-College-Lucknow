import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction using Linear Regression")

uploaded_file = st.file_uploader(
    "Upload homeprices.csv", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    X = df[['area']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    st.success("Model trained successfully!")

    st.subheader("Predict House Price")

    area = st.number_input(
        "Enter Area (sq ft)",
        min_value=100,
        value=3300,
        step=100
    )

    if st.button("Predict Price"):
        prediction = model.predict([[area]])[0]

        st.metric(
            label="Predicted House Price",
            value=f"${prediction:,.2f}"
        )

    st.subheader("Model Information")
    st.write(f"Coefficient (m): {model.coef_[0]:.4f}")
    st.write(f"Intercept (b): {model.intercept_:.4f}")

else:
    st.info("Please upload the homeprices.csv dataset.")
