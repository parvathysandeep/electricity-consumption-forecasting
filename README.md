# ⚡ Electricity Consumption Forecasting System
### Deep Learning (Stacked LSTM) vs. Traditional Machine Learning Models

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-yellow.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

An end-to-end, reproducible Machine Learning and Deep Learning project developed for academic evaluation and energy demand forecasting. The system analyzes **12 continuous years (4,383 daily records)** of real-world grid electricity consumption from the Open Power System Data (OPSD) platform, implements a zero-data-leakage pipeline, benchmarks traditional models against a stacked LSTM recurrent neural network, and serves forecasts via both a Command Line Interface (CLI) and an interactive Streamlit web application.

---

## 📋 Table of Contents

1. [Executive Summary & Benchmark Results](#-executive-summary--benchmark-results)
2. [System Prerequisites](#-system-prerequisites)
3. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
   - [Step 1: Open Terminal in Project Directory](#step-1-open-terminal-in-project-directory)
   - [Step 2: (Optional) Set Up a Virtual Environment](#step-2-optional-set-up-a-virtual-environment)
   - [Step 3: Install Required Dependencies](#step-3-install-required-dependencies)
   - [Step 4: Execute the Training & Evaluation Pipeline](#step-4-execute-the-training--evaluation-pipeline)
   - [Step 5: Launch the Interactive Streamlit Web App](#step-5-launch-the-interactive-streamlit-web-app)
   - [Step 6: Run Command-Line Forecasts](#step-6-run-command-line-forecasts)
   - [Step 7: Run the Automated Unit Test Suite](#step-7-run-the-automated-unit-test-suite)
4. [Project Architecture & File Catalog](#-project-architecture--file-catalog)
5. [Generated Output Plots & Artifacts](#-generated-output-plots--artifacts)
6. [Methodology & Zero-Data-Leakage Guarantee](#-methodology--zero-data-leakage-guarantee)
7. [Academic Submission Deliverables](#-academic-submission-deliverables)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 📊 Executive Summary & Benchmark Results

All models were evaluated on the exact same **held-out test set (656 days: March 16, 2016 – December 31, 2017)**. Test metrics are reported in original physical units (Gigawatt-hours, GWh):

| Model Architecture | MAE (GWh) | RMSE (GWh) | MAPE (%) | $R^2$ Score | Performance Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **24.942** | **45.941** | **1.908%** | **0.9178** | 🥇 **1st Place** |
| **Linear Regression (Ridge)** | 34.047 | 55.959 | 2.580% | 0.8781 | 🥈 **2nd Place** |
| **Stacked LSTM Network** | 37.964 | 64.306 | 2.913% | 0.8390 | 🥉 **3rd Place** |
| **Persistence (Naive Lag-1)** | 102.040 | 149.474 | 7.761% | 0.1300 | 4th (Baseline) |

### Key Scientific Takeaways
- **Over 60% Error Reduction**: All machine learning and deep learning models dramatically outperform the naive baseline ($MAE = 102.04\text{ GWh} \to 24.94\text{ GWh}$).
- **Random Forest Strength**: Outperforms on daily data due to explicit calendar indicators (`day_of_week`, `is_weekend`, lag-7) that cleanly capture the sharp drop during weekend industrial cycles.
- **LSTM Deep Representation**: The stacked LSTM network achieves a **2.91% MAPE purely from raw 14-day sequence inputs** without manual feature engineering, proving its gating mechanism naturally extracts seasonal rhythms.

---

## 💻 System Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python Version**: Python 3.9, 3.10, or 3.11.
- **Hardware**: Standard laptop CPU (GPU is completely optional; training takes under 45 seconds on CPU).
- **Disk Space**: ~500 MB (includes dataset, Python libraries, and saved weights).

---

## 🚀 Step-by-Step Execution Guide

Follow these sequential steps to set up, train, evaluate, and demonstrate the project.

### Step 1: Open Terminal in Project Directory

Open **PowerShell**, **Command Prompt**, or **Bash** and navigate to your project directory:

```bash
cd "c:\Users\DELL\Desktop\AI PROJECT"
```

Verify that you are in the correct directory:
```bash
python --version
```

---

### Step 2: (Optional) Set Up a Virtual Environment

Creating an isolated virtual environment is recommended to avoid version conflicts:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies

Install all verified dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Packages Installed**: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `joblib`, and `streamlit`.

---

### Step 4: Execute the Training & Evaluation Pipeline

Run the master orchestrator script `main.py`:

```bash
python main.py
```

#### What happens during execution:
1. **Data Acquisition**: Checks for `data/electricity_consumption.csv`. If missing, automatically downloads the real 12-year OPSD Germany dataset.
2. **Exploratory Visualizations**: Saves 12-year consumption overview to `outputs/1_consumption_time_series.png`.
3. **Feature Engineering**: Builds cyclical calendar encodings, historical lag features ($t-1, t-2, t-7, t-14$), and shifted 7-day rolling statistics.
4. **Leakage-Free Splitting**: Chronologically partitions data into 70% Train, 15% Validation, and 15% Test. Fits `MinMaxScaler` **only** on the training set and saves `models/scaler.joblib`.
5. **Baseline Training**: Fits Naive Persistence, Ridge Linear Regression, and a 150-tree Random Forest (`models/random_forest_model.joblib`).
6. **LSTM Deep Learning**: Trains a stacked 2-layer LSTM model with Dropout ($p=0.2$) and EarlyStopping. Best checkpoint saved to `models/best_lstm_model.keras`.
7. **Test Evaluation**: Predicts on 656 held-out test days; calculates MAE, RMSE, MAPE, and $R^2$; outputs `outputs/metrics_summary.csv`.
8. **Plot Generation**: Generates 5 high-resolution evaluation figures in `outputs/`.

---

### Step 5: Launch the Interactive Streamlit Web App

To demonstrate the working model in a graphical interface:

```bash
streamlit run app.py
```

Your browser will automatically open:
```text
Local URL: http://localhost:8501
```

#### What you can do inside the web app:
- **📊 Project Dashboard & Overview**: Explore historical consumption curves and weekly/monthly seasonality boxplots.
- **📈 Model Evaluation & Comparison**: View interactive metric tables and side-by-side error bar charts.
- **🔮 Live Interactive Forecasting**: Adjust sliders for **Lookback Window (7–30 days)** and **Forecast Horizon (1–14 days ahead)** to watch the trained LSTM neural network predict future grid load in real time.
- **📚 Model Architecture & Viva Notes**: Review mathematical LSTM gate equations and data leakage prevention principles for project defense.

*(To stop the app, press `Ctrl + C` in the terminal).*

---

### Step 6: Run Command-Line Forecasts

Generate predictions directly from the command line using `src/predict.py`:

#### Option A: Predict Next-Day Electricity Demand (Single-Step)
```bash
python src/predict.py --steps 14 --horizon 1
```
**Example Output:**
```text
=============================================
 FORECAST RESULT: 1221.95 GWh
 Input Mean:     1321.80 GWh
=============================================
```

#### Option B: Predict 7 Days Ahead (Autoregressive Multi-Step Rollout)
```bash
python src/predict.py --steps 14 --horizon 7
```
**Example Output:**
```text
=============================================
 7-DAY AHEAD FORECAST (GWh):
 Day +1: 1221.95 GWh
 Day +2: 1326.67 GWh
 Day +3: 1352.09 GWh
 Day +4: 1372.27 GWh
 Day +5: 1380.97 GWh
 Day +6: 1245.30 GWh
 Day +7: 1120.15 GWh
=============================================
```

---

### Step 7: Run the Automated Unit Test Suite

Verify system integrity, sequence dimensions, metric calculations, and validation rules:

```bash
python -m unittest tests/test_pipeline.py
```

**Expected Result:**
```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.091s

OK
```

---

## 🗂️ Project Architecture & File Catalog

```text
AI PROJECT/
├── data/
│   ├── electricity_consumption.csv   # Real OPSD German grid dataset (4,383 daily records)
│   └── README.md                     # Dataset documentation and column schema
├── models/
│   ├── best_lstm_model.keras         # Trained Keras stacked LSTM neural network checkpoint
│   ├── random_forest_model.joblib    # Serialized scikit-learn Random Forest model
│   ├── scaler.joblib                 # MinMaxScaler fit exclusively on training data
│   ├── model_metadata.json           # Model configuration, hyperparameters, and results
│   └── README.md                     # Model artifacts inventory
├── notebooks/
│   └── exploratory_analysis.ipynb    # Jupyter notebook for interactive EDA and experimentation
├── outputs/
│   ├── 1_consumption_time_series.png # 12-year time-series trend & 30-day moving average
│   ├── 2_training_validation_loss.png# LSTM training and validation loss curves
│   ├── 3_actual_vs_predicted_test.png# Multi-model actual vs predicted overlay on test set
│   ├── 4_residual_distribution.png   # Residual error distributions (KDE + histogram)
│   ├── 5_model_comparison_metrics.png# Comparative bar charts for MAE, RMSE, and MAPE
│   ├── metrics_summary.csv           # Tabular metrics in CSV format
│   ├── metrics_summary.json          # Formatted metrics in JSON format
│   └── README.md                     # Visualizations catalog
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_loader.py                # Dataset download, validation, cleaning, and indexing
│   ├── feature_engineering.py        # Calendar encodings, lags, and shifted rolling statistics
│   ├── preprocessing.py              # Chronological splitting, scaling, and 3D sequence creation
│   ├── baseline_models.py            # Persistence, Ridge regression, and Random Forest models
│   ├── lstm_model.py                 # Stacked LSTM architecture, early stopping, and training
│   ├── evaluation.py                 # MAE, RMSE, MAPE, and R2 calculation routines
│   ├── visualization.py              # Publication-quality plotting utilities
│   └── predict.py                    # Standalone inference engine and autoregressive rollout
├── tests/
│   └── test_pipeline.py              # Automated test suite (4 unit/integration tests)
├── app.py                            # Interactive Streamlit dashboard
├── main.py                           # Master end-to-end execution pipeline
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation and user guide
├── report.md                         # ~10-page academic report with IEEE references
├── presentation.md                   # 10-slide seminar presentation deck
└── viva_prep.md                      # 15+ viva defense questions and model answers
```

---

## 📈 Generated Output Plots & Artifacts

All figures are automatically generated by `python main.py` at 300 DPI:

1. **`outputs/1_consumption_time_series.png`**: Complete historical consumption record showing yearly winter peaks and long-term grid stability.
2. **`outputs/2_training_validation_loss.png`**: Training vs. validation loss curve confirming convergence without overfitting (EarlyStopping restored best weights).
3. **`outputs/3_actual_vs_predicted_test.png`**: 120-day zoom of the held-out test partition demonstrating how Random Forest and LSTM accurately anticipate weekend load drops.
4. **`outputs/4_residual_distribution.png`**: Error distributions showing zero-centered, low-variance Gaussian residuals for ML models versus wide heavy tails for naive persistence.
5. **`outputs/5_model_comparison_metrics.png`**: Side-by-side bar chart benchmarking MAE, RMSE, and MAPE across all four approaches.

---

## 🛡️ Methodology & Zero-Data-Leakage Guarantee

Time-series forecasting requires strict protocols to prevent lookahead bias:

1. **Strict Chronological Splitting**:
   - **Training Set (70%)**: 2006-01-01 to 2014-05-18 (3,068 days)
   - **Validation Set (15%)**: 2014-05-19 to 2016-03-15 (657 days)
   - **Test Set (15%)**: 2016-03-16 to 2017-12-31 (656 days)
2. **Isolated Scaling**: The `MinMaxScaler` is fit **only on the training split**. Validation and test data are scaled using the pre-computed training parameters.
3. **Shifted Rolling Windows**: All rolling window averages are calculated with `.shift(1)`, guaranteeing that day $t$'s demand is never used as a predictor for day $t$.
4. **Physical Scale Inversion**: All predictions are converted back to GWh before calculating MAE, RMSE, and MAPE.

---

## 🎓 Academic Submission Deliverables

The repository includes complete submission materials:

- 📄 **[Academic Report (`report.md`)](report.md)**: ~10-page academic report complete with abstract, literature review, mathematical derivations of LSTM gates ($f_t, i_t, \tilde{C}_t, C_t, o_t, h_t$), empirical results, and IEEE citations.
- 🖥️ **[Presentation Deck (`presentation.md`)](presentation.md)**: 10 structured slides formatted for project viva and seminar presentations.
- 🎯 **[Viva Preparation Guide (`viva_prep.md`)](viva_prep.md)**: 15+ viva questions with concise answers explaining model selection, vanishing gradients, error metrics, and trade-offs.

---

## ❓ Troubleshooting & FAQ

#### Q1: "Port 8501 is already in use when running Streamlit"
**Fix**: Specify a different port:
```bash
streamlit run app.py --server.port 8502
```

#### Q2: "Can I run this offline without internet access?"
**Fix**: Yes. The dataset has already been downloaded and cached at `data/electricity_consumption.csv`. The pipeline will use the cached file automatically.

#### Q3: "Can I use my own custom hourly or daily dataset?"
**Fix**: Yes. Place any CSV inside `data/` containing a timestamp column (`Date` or `Datetime`) and a demand column (`Consumption`, `Load`, `Demand`, or `PJME_MW`). The pipeline automatically detects the schema.
