"""
Main End-to-End Orchestrator for Electricity Consumption Forecasting.
Executes the full machine learning and deep learning lifecycle:
1. Data Acquisition & Validation
2. Feature Engineering & Exploratory Visualizations
3. Chronological Preprocessing & Scaling (Zero Data Leakage)
4. Baseline Model Training (Persistence, Ridge, Random Forest)
5. LSTM Neural Network Training (EarlyStopping & Checkpointing)
6. Test Set Evaluation (MAE, RMSE, MAPE, R2)
7. Publication Visualizations & Model Artifact Serialization
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Fix seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

from src.data_loader import load_electricity_data
from src.feature_engineering import build_feature_pipeline
from src.preprocessing import prepare_lstm_data, prepare_tabular_data, inverse_transform_predictions
from src.baseline_models import evaluate_baselines
from src.lstm_model import train_lstm, DEFAULT_LSTM_PATH
from src.evaluation import compare_models, OUTPUT_DIR
from src.visualization import (
    plot_consumption_time_series,
    plot_training_validation_loss,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison_bar
)


def run_pipeline(
    n_steps: int = 14,
    epochs: int = 35,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    print("=" * 70)
    print(" ELECTRICITY DEMAND FORECASTING PIPELINE: LSTM vs. BASELINE MODELS")
    print("=" * 70)

    # 1. Dataset Acquisition & Validation
    print("\n--- STAGE 1: Dataset Acquisition & Validation ---")
    raw_df = load_electricity_data()

    # 2. Exploratory Visualization
    print("\n--- STAGE 2: Exploratory Data Analysis & Trends ---")
    plot_consumption_time_series(raw_df)

    # 3. Feature Engineering for Baselines
    print("\n--- STAGE 3: Feature Engineering ---")
    feature_df = build_feature_pipeline(raw_df)

    # 4. Preprocessing & Dataset Splitting
    print("\n--- STAGE 4: Chronological Splitting & Scaling ---")
    lstm_data = prepare_lstm_data(
        df=raw_df,
        n_steps=n_steps,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    tabular_data = prepare_tabular_data(
        feature_df=feature_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    # 5. Baseline Models Training & Evaluation
    print("\n--- STAGE 5: Baseline Models Evaluation ---")
    baseline_results = evaluate_baselines(tabular_data)

    # 6. LSTM Model Training
    print("\n--- STAGE 6: LSTM Neural Network Training ---")
    lstm_model, history = train_lstm(
        X_train=lstm_data["X_train"],
        y_train=lstm_data["y_train"],
        X_val=lstm_data["X_val"],
        y_val=lstm_data["y_val"],
        epochs=epochs,
        batch_size=batch_size,
        patience=8
    )

    # Plot LSTM Training/Validation Convergence
    plot_training_validation_loss(history)

    # 7. LSTM Prediction on Test Set
    print("\n--- STAGE 7: LSTM Evaluation on Held-Out Test Data ---")
    y_test_scaled_pred = lstm_model.predict(lstm_data["X_test"], verbose=0)
    lstm_pred_unscaled = inverse_transform_predictions(lstm_data["scaler"], y_test_scaled_pred)

    # Exact common test date alignment across LSTM and Tabular Baselines
    lstm_test_dates = lstm_data["test_dates"]
    tab_test_dates = tabular_data["test_dates"]
    
    # Common timestamps intersection
    common_dates = [d for d in lstm_test_dates if d in tab_test_dates]
    
    lstm_mask = [i for i, d in enumerate(lstm_test_dates) if d in common_dates]
    tab_mask = [i for i, d in enumerate(tab_test_dates) if d in common_dates]

    actual_test = lstm_data["test_actual_unscaled"][lstm_mask]
    lstm_pred_aligned = lstm_pred_unscaled[lstm_mask]

    aligned_predictions = {
        "Persistence": baseline_results["Persistence"][tab_mask],
        "Linear Regression": baseline_results["Linear Regression"][tab_mask],
        "Random Forest": baseline_results["Random Forest"][tab_mask],
        "LSTM Network": lstm_pred_aligned
    }

    print(f"[Pipeline] Aligned all models on {len(actual_test)} identical test dates ({common_dates[0].date()} to {common_dates[-1].date()}).")

    # 8. Evaluation Metrics Comparison
    print("\n--- STAGE 8: Model Performance Comparison ---")
    eval_dict = {}
    for name, preds in aligned_predictions.items():
        eval_dict[name] = {"actual": actual_test, "predicted": preds}

    df_summary = compare_models(eval_dict)

    # 9. Visualizations
    print("\n--- STAGE 9: Generating Comparative Evaluation Visualizations ---")
    plot_actual_vs_predicted(pd.DatetimeIndex(common_dates), actual_test, aligned_predictions)
    plot_residuals(actual_test, aligned_predictions)
    plot_model_comparison_bar(df_summary)

    # Save model metadata
    metadata_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")
    metadata = {
        "dataset_name": "Open Power System Data (Germany Daily Consumption)",
        "target_variable": "Consumption (GWh)",
        "lookback_window_n_steps": n_steps,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": batch_size,
        "train_samples": len(lstm_data["X_train"]),
        "val_samples": len(lstm_data["X_val"]),
        "test_samples": len(actual_test),
        "metrics_summary": df_summary.to_dict(orient="records")
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Pipeline] Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df_summary


if __name__ == "__main__":
    run_pipeline()
