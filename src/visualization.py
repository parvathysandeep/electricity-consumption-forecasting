"""
Visualization Module for Electricity Demand Forecasting.
Generates publication-ready figures for EDA, model convergence,
actual vs predicted forecasts, residual distributions, and metric comparisons.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_consumption_time_series(df: pd.DataFrame, target_col: str = "Consumption", save_dir: str = OUTPUT_DIR) -> str:
    """
    1. Plots full historical time-series of electricity demand with 30-day rolling trend.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df[target_col], label="Daily Consumption (GWh)", color="#4A90E2", alpha=0.55, linewidth=1.0)
    
    # 30-day moving average
    rolling_30 = df[target_col].rolling(window=30).mean()
    ax.plot(df.index, rolling_30, label="30-Day Moving Average", color="#D0021B", linewidth=2.0)

    ax.set_title("Historical Electricity Consumption in Germany (2006 - 2017)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Electricity Demand (GWh)", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_file = os.path.join(save_dir, "1_consumption_time_series.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[Visualization] Saved: {out_file}")
    return out_file


def plot_training_validation_loss(history, save_dir: str = OUTPUT_DIR) -> str:
    """
    2. Plots LSTM training vs validation loss across epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"], label="Training Loss (MSE)", color="#204051", linewidth=2.2)
    ax.plot(epochs, history.history["val_loss"], label="Validation Loss (MSE)", color="#FF8C00", linewidth=2.2, linestyle="--")

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_loss = min(history.history["val_loss"])
    ax.scatter([best_epoch], [best_loss], color="red", s=80, zorder=5, label=f"Best Val Loss ({best_loss:.5f} @ Epoch {best_epoch})")

    ax.set_title("LSTM Neural Network Convergence: Training vs. Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_file = os.path.join(save_dir, "2_training_validation_loss.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[Visualization] Saved: {out_file}")
    return out_file


def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predictions_dict: dict,
    zoom_days: int = 120,
    save_dir: str = OUTPUT_DIR
) -> str:
    """
    3. Plots actual demand vs predicted demand for all models on test set (zoomed to last N days for clarity).
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 6))

    # Zoom window for granular clarity
    dates_zoom = dates[-zoom_days:]
    actual_zoom = actual[-zoom_days:]

    ax.plot(dates_zoom, actual_zoom, label="Actual Consumption", color="black", linewidth=2.2, zorder=4)

    colors = {"Persistence": "#9B59B6", "Linear Regression": "#3498DB", "Random Forest": "#2ECC71", "LSTM Network": "#E74C3C"}
    styles = {"Persistence": ":", "Linear Regression": "-.", "Random Forest": "--", "LSTM Network": "-"}

    for name, preds in predictions_dict.items():
        preds_zoom = preds[-zoom_days:]
        ax.plot(dates_zoom, preds_zoom, label=name, color=colors.get(name, None),
                linestyle=styles.get(name, "-"), linewidth=1.8, alpha=0.85)

    ax.set_title(f"Test Set Electricity Demand Forecast Comparison (Last {zoom_days} Days)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Electricity Demand (GWh)", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_file = os.path.join(save_dir, "3_actual_vs_predicted_test.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[Visualization] Saved: {out_file}")
    return out_file


def plot_residuals(actual: np.ndarray, predictions_dict: dict, save_dir: str = OUTPUT_DIR) -> str:
    """
    4. Plots residual error distribution (Actual - Predicted) for all evaluated models.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {"Persistence": "#9B59B6", "Linear Regression": "#3498DB", "Random Forest": "#2ECC71", "LSTM Network": "#E74C3C"}

    for name, preds in predictions_dict.items():
        residuals = actual - preds
        sns.kdeplot(residuals, label=f"{name} (Mean={residuals.mean():.1f}, Std={residuals.std():.1f})",
                    ax=ax, linewidth=2.0, color=colors.get(name, None))

    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("Forecast Residual Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Residual Error (GWh)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_file = os.path.join(save_dir, "4_residual_distribution.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[Visualization] Saved: {out_file}")
    return out_file


def plot_model_comparison_bar(df_summary: pd.DataFrame, save_dir: str = OUTPUT_DIR) -> str:
    """
    5. Generates grouped bar chart comparing MAE, RMSE, and MAPE across models.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot MAE & RMSE
    df_melt_err = df_summary.melt(id_vars=["Model"], value_vars=["MAE (GWh)", "RMSE (GWh)"], var_name="Metric", value_name="Error (GWh)")
    sns.barplot(data=df_melt_err, x="Model", y="Error (GWh)", hue="Metric", ax=ax1, palette="Blues_d")
    ax1.set_title("Forecast Absolute & Squared Error (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Error (GWh)")
    ax1.tick_params(axis='x', rotation=15)
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f"{height:.1f}", (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # Plot MAPE (%)
    sns.barplot(data=df_summary, x="Model", y="MAPE (%)", ax=ax2, palette="Oranges_d")
    ax2.set_title("Mean Absolute Percentage Error (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MAPE (%)")
    ax2.tick_params(axis='x', rotation=15)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    out_file = os.path.join(save_dir, "5_model_comparison_metrics.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[Visualization] Saved: {out_file}")
    return out_file
