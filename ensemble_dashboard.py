import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="Ensemble Forecast Viewer", layout="wide")
st.title("📈 LSTM + XGBoost Ensemble Forecast")
st.markdown("Upload a CSV containing columns: `actual`, `lstm`, `xgb`, and `ensemble`. No `date` column needed.")

uploaded_file = st.file_uploader("Upload `streamlit_input.csv`", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ['actual', 'lstm', 'xgb', 'ensemble']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f" Missing required columns: {missing_cols}")
        else:
            # Metrics
            mae = mean_absolute_error(df['actual'], df['ensemble'])
            mse = mean_squared_error(df['actual'], df['ensemble'])
            mape = np.mean(np.abs((df['actual'] - df['ensemble']) / df['actual'])) * 100

            st.subheader("📊 Forecast Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.4f}")
            col2.metric("MSE", f"{mse:.4f}")
            col3.metric("MAPE", f"{mape:.2f}%")

            # Plot using index as x-axis
            st.subheader("📉 Forecast Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['actual'].values, label='Actual', color='black')
            ax.plot(df['lstm'].values, label='LSTM', linestyle='--', color='purple', alpha=0.5)
            ax.plot(df['xgb'].values, label='XGBoost', linestyle='--', color='green', alpha=0.5)
            ax.plot(df['ensemble'].values, label='Ensemble', color='red', linewidth=2)
            ax.set_xlabel('Index')
            ax.set_ylabel('Target Value')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.success("✅ Forecast loaded and visualised successfully.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
