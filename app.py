"""
Interactive Streamlit Web Dashboard for Electricity Consumption Forecasting.
Provides interactive model evaluation, comparative visualizations,
and live future forecasting using trained LSTM and baseline models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to python path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_electricity_data
from src.preprocessing import load_scaler, prepare_lstm_data, inverse_transform_predictions, DEFAULT_SCALER_PATH
from src.lstm_model import load_trained_lstm, DEFAULT_LSTM_PATH
from src.predict import predict_next_step, predict_multi_step_autoregressive
from src.evaluation import calculate_metrics

# Page configuration
st.set_page_config(
    page_title="Electricity Demand Forecasting | LSTM & ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Electricity Demand Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparative Analysis of Deep Learning (LSTM) vs. Machine Learning Baselines</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/electricity.png", width=70)
st.sidebar.title("Navigation & Controls")
nav_option = st.sidebar.radio(
    "Select View:",
    ["📊 Project Dashboard & Overview", "📈 Model Evaluation & Comparison", "🔮 Live Interactive Forecasting", "📚 Model Architecture & Viva Notes"]
)

# Cache data loading
@st.cache_data
def get_data():
    return load_electricity_data()

try:
    df_raw = get_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure data is present or run main.py first.")
    st.stop()

# VIEW 1: PROJECT OVERVIEW
if nav_option == "📊 Project Dashboard & Overview":
    st.header("Project Overview & Dataset Exploratory Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df_raw):,}")
    with col2:
        st.metric("Date Range", f"{df_raw.index.min().year} - {df_raw.index.max().year}")
    with col3:
        st.metric("Mean Demand", f"{df_raw['Consumption'].mean():.1f} GWh")
    with col4:
        st.metric("Peak Demand", f"{df_raw['Consumption'].max():.1f} GWh")

    st.subheader("Historical Electricity Consumption")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_raw.index, df_raw["Consumption"], label="Daily Consumption", color="#3B82F6", alpha=0.5, linewidth=1)
    rolling_30 = df_raw["Consumption"].rolling(30).mean()
    ax.plot(df_raw.index, rolling_30, label="30-Day Moving Average", color="#EF4444", linewidth=2)
    ax.set_ylabel("Demand (GWh)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Seasonal Patterns (Weekly & Monthly)")
    c1, c2 = st.columns(2)
    with c1:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        df_dow = df_raw.copy()
        df_dow["Day"] = [day_names[i] for i in df_dow.index.dayofweek]
        fig_dow, ax_dow = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_dow, x="Day", y="Consumption", palette="Blues", ax=ax_dow)
        ax_dow.set_title("Weekly Consumption Profile (Drop on Weekends)")
        st.pyplot(fig_dow)
        plt.close(fig_dow)

    with c2:
        df_month = df_raw.copy()
        df_month["Month"] = df_month.index.month
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_month, x="Month", y="Consumption", palette="Oranges", ax=ax_m)
        ax_m.set_title("Monthly Consumption Profile (Winter Peaks)")
        st.pyplot(fig_m)
        plt.close(fig_m)

# VIEW 2: MODEL EVALUATION & COMPARISON
elif nav_option == "📈 Model Evaluation & Comparison":
    st.header("Comparative Model Performance on Held-Out Test Set")

    metrics_csv = os.path.join(os.path.dirname(__file__), "outputs", "metrics_summary.csv")
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv)
        st.dataframe(df_metrics.style.highlight_min(subset=["MAE (GWh)", "RMSE (GWh)", "MAPE (%)"], color="#D1FAE5"), use_container_width=True)

        st.subheader("Benchmark Comparison Charts")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_metrics, x="Model", y="RMSE (GWh)", palette="Blues_r", ax=ax)
            ax.set_title("Root Mean Squared Error (RMSE - Lower is Better)")
            plt.xticks(rotation=20)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_metrics, x="Model", y="MAPE (%)", palette="Oranges_r", ax=ax)
            ax.set_title("Mean Absolute Percentage Error (MAPE %)")
            plt.xticks(rotation=20)
            st.pyplot(fig)
            plt.close(fig)

        # Show actual saved plots if present
        img_pred = os.path.join(os.path.dirname(__file__), "outputs", "3_actual_vs_predicted_test.png")
        if os.path.exists(img_pred):
            st.subheader("Actual vs Predicted Electricity Demand (Zoomed Test Period)")
            st.image(img_pred, use_container_width=True)

        img_loss = os.path.join(os.path.dirname(__file__), "outputs", "2_training_validation_loss.png")
        if os.path.exists(img_loss):
            st.subheader("LSTM Convergence History")
            st.image(img_loss, use_container_width=True)
    else:
        st.warning("Metrics not found. Please run `python main.py` first to train models and generate evaluation metrics.")

# VIEW 3: LIVE FORECASTING
elif nav_option == "🔮 Live Interactive Forecasting":
    st.header("Interactive Forecasting Demonstration")
    st.write("Generate real-time electricity forecasts by selecting recent historical context.")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        horizon = st.slider("Forecasting Horizon (Days Ahead):", min_value=1, max_value=14, value=7, step=1)
    with col_ctrl2:
        lookback = st.slider("Lookback History Window (Days):", min_value=7, max_value=30, value=14, step=1)

    # Use the latest observations from the dataset
    recent_series = df_raw["Consumption"].tail(lookback).values
    
    st.write(f"**Last {lookback} Days Input Observations (GWh):**")
    st.write([round(float(x), 1) for x in recent_series])

    if st.button("Generate Forecast 🚀", type="primary"):
        with st.spinner("Executing neural network forward pass..."):
            try:
                preds = predict_multi_step_autoregressive(
                    recent_observations=recent_series,
                    horizon=horizon,
                    n_steps=14 # Default model lookback
                )

                st.success(f"Forecast successfully computed for next {horizon} day(s)!")
                
                # Visual display
                forecast_dates = pd.date_range(start=df_raw.index.max() + pd.Timedelta(days=1), periods=horizon)
                df_forecast = pd.DataFrame({
                    "Date": forecast_dates.strftime("%Y-%m-%d"),
                    "Day of Week": forecast_dates.day_name(),
                    "Predicted Demand (GWh)": preds
                })
                
                c_tbl, c_plot = st.columns([1, 2])
                with c_tbl:
                    st.dataframe(df_forecast, use_container_width=True)
                with c_plot:
                    fig_fc, ax_fc = plt.subplots(figsize=(8, 4))
                    hist_dates = df_raw.index[-lookback:]
                    ax_fc.plot(hist_dates, recent_series, label="Historical Input", color="black", marker="o")
                    ax_fc.plot(forecast_dates, preds, label="LSTM Forecast", color="#E11D48", linestyle="--", marker="s")
                    ax_fc.set_ylabel("Demand (GWh)")
                    ax_fc.set_title(f"Next {horizon}-Day Demand Forecast")
                    ax_fc.legend()
                    st.pyplot(fig_fc)
                    plt.close(fig_fc)

            except Exception as e:
                st.error(f"Inference error: {e}. Ensure models are trained by running `python main.py`.")

# VIEW 4: VIVA & ARCHITECTURE NOTES
else:
    st.header("LSTM Mathematical Foundation & Academic Project Defense")
    
    st.markdown("""
    ### 1. Long Short-Term Memory (LSTM) Architecture
    Standard RNNs suffer from **vanishing and exploding gradients** over long sequences.
    LSTM solves this with a **Cell State ($C_t$)** acting as an information conveyor belt regulated by three gates:
    
    - **Forget Gate ($f_t$)**: Decides what percentage of old cell memory to discard.
      $$\\mathbf{f}_t = \\sigma(\\mathbf{W}_f \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_f)$$
    
    - **Input Gate ($i_t$)**: Determines which new values will be updated in the cell state.
      $$\\mathbf{i}_t = \\sigma(\\mathbf{W}_i \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_i)$$
      $$\\tilde{\\mathbf{C}}_t = \\tanh(\\mathbf{W}_c \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_c)$$
    
    - **Cell State Update ($C_t$)**: Combines forgotten past with scaled candidate input.
      $$\\mathbf{C}_t = \\mathbf{f}_t \\odot \\mathbf{C}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{C}}_t$$
    
    - **Output Gate ($o_t$) & Hidden State ($h_t$)**: Selectively filters cell memory to produce output.
      $$\\mathbf{o}_t = \\sigma(\\mathbf{W}_o \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_o)$$
      $$\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{C}_t)$$

    ---
    ### 2. Prevention of Data Leakage
    - **Chronological Splitting**: Shuffling time-series observations corrupts temporal causality. Split 70% Train, 15% Val, 15% Test strictly by timestamp.
    - **Scaler Fitting**: MinMaxScaler is fit **only** on the training partition. The validation and test partitions are transformed using the training parameters.
    - **Rolling Window Shifts**: All lag and rolling statistics are shifted by at least 1 day ($t-1$) to guarantee future information is never accessible during prediction.
    """)
