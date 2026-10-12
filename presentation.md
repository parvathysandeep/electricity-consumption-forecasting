# Electricity Consumption Forecasting: LSTM vs. Traditional ML Models
## Academic Seminar & Project Defense Slide Deck (10 Slides)

---

### Slide 1: Title & Project Overview
- **Project Title**: Electricity Consumption Forecasting Using LSTM Neural Networks and Traditional Machine Learning Models
- **Course**: Machine Learning & Applied Artificial Intelligence (10 Marks Case Study)
- **Domain**: Time-Series Forecasting & Smart Grid Energy Dispatch
- **Key Architectures**: Persistence (Naive), Ridge Linear Regression, Random Forest, Stacked LSTM
- **Deliverables**: End-to-end Python pipeline, serialized models, publication plots, Streamlit web demo

---

### Slide 2: Problem Statement & Motivation
- **The Core Problem**: Electricity cannot be stored cheaply at grid scale; generation must match load in real-time.
- **Consequences of Forecasting Errors**:
  - Over-prediction: Unnecessary spinning reserves, fossil fuel burning, excessive operational costs.
  - Under-prediction: Frequency drops, brownouts, expensive spot market purchases, grid collapse.
- **The Challenge**: Electricity demand exhibits complex multi-scale seasonality:
  - Weekly periodicity: Distinct industrial drops on weekends.
  - Annual seasonality: High winter demand peaks vs. mild summer troughs.

---

### Slide 3: Project Objectives
1. **Acquire Public Data**: Ingest 12 years of real-world German national electricity load data (Open Power System Data).
2. **Prevent Data Leakage**: Enforce strict chronological train/val/test splitting and isolated scaler fitting.
3. **Develop Diverse Baselines**: Establish Persistence, Ridge, and Random Forest benchmark models.
4. **Implement Deep LSTM**: Build a stacked LSTM recurrent neural network in TensorFlow/Keras.
5. **Rigorous Empirical Comparison**: Evaluate on identical held-out test partitions using MAE, RMSE, MAPE, and $R^2$.
6. **Provide Working Demo**: Build interactive CLI and Streamlit interfaces for live predictions.

---

### Slide 4: Dataset & Exploratory Data Analysis (EDA)
- **Source**: Open Power System Data (OPSD) platform (Germany national grid).
- **Time Span**: 2006-01-01 to 2017-12-31 (12 full calendar years).
- **Total Records**: 4,383 daily observations (0 missing values in target).
- **Target Variable**: `Consumption` (Total daily electricity demand in GWh).
  - Mean: $1,338.68\text{ GWh}$, Std Dev: $165.78\text{ GWh}$.
  - Min: $842.39\text{ GWh}$, Max: $1,709.57\text{ GWh}$.
- **Key EDA Findings**:
  - Weekday demand is 20–30% higher than Saturday/Sunday demand.
  - Winter months (Dec–Feb) peak due to electric heating and lighting.

---

### Slide 5: Data Preprocessing & Zero-Leakage Pipeline
- **Why Random Splitting Fails**: Random shuffling introduces *lookahead bias*—future observations leak into the training past.
- **Chronological Partitioning**:
  - **Train (70%)**: 2006 to May 2014 ($N = 3,068$)
  - **Validation (15%)**: May 2014 to March 2016 ($N = 657$)
  - **Test (15%)**: March 2016 to December 2017 ($N = 656$)
- **Zero-Leakage Rules**:
  - `MinMaxScaler` fit strictly on Training Set; Validation and Test only transformed.
  - All rolling mean/std statistics shifted by $t-1$.
  - 3D sliding window tensors: $[N, 14, 1]$ mapped to single-step target $y_{t+1}$.

---

### Slide 6: LSTM Architecture & Gate Equations
- **Why LSTM?**: Resolves vanishing gradients in standard RNNs via gated memory cell state ($C_t$).
- **Gate Operations**:
  - **Forget Gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ (discards irrelevant past).
  - **Input Gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ and $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ (adds new info).
  - **Cell Update**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ (linear conveyor belt).
  - **Output Gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$; $h_t = o_t \odot \tanh(C_t)$.
- **Network Configuration**:
  - Input: $(14, 1) \rightarrow \text{LSTM}(64) \rightarrow \text{Dropout}(0.2) \rightarrow \text{LSTM}(32) \rightarrow \text{Dropout}(0.2) \rightarrow \text{Dense}(16) \rightarrow \text{Dense}(1)$.
  - Adam Optimizer ($\text{lr}=0.001$), Loss: MSE, EarlyStopping ($P=8$).

---

### Slide 7: Baseline Machine Learning Models
- **Model 1: Naive Persistence**:
  - $\hat{y}_{t+1} = y_t$. Serves as the minimal sanity check.
- **Model 2: Ridge Linear Regression**:
  - Regularized linear formulation on calendar and lag features ($\alpha = 1.0$).
- **Model 3: Random Forest Regressor**:
  - Ensemble of 150 randomized decision trees on calendar identifiers, cyclical sine/cosine encodings, autoregressive lags ($t-1, t-2, t-7, t-14$), and 7-day rolling statistics.

---

### Slide 8: Experimental Evaluation & Results
**Evaluated on 656 days held-out test data (March 2016 – December 2017):**

| Model | MAE (GWh) | RMSE (GWh) | MAPE (%) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **24.942** | **45.941** | **1.908%** | **0.9178** | **1st** |
| **Linear Regression (Ridge)** | 34.047 | 55.959 | 2.580% | 0.8781 | 2nd |
| **Stacked LSTM Network** | 37.964 | 64.306 | 2.913% | 0.8390 | 3rd |
| **Persistence (Naive Lag-1)** | 102.040 | 149.474 | 7.761% | 0.1300 | 4th |

- **Key Takeaway**: All learning models cut naive error by more than **60%**, achieving sub-3% error rates.

---

### Slide 9: Practical Applications & Limitations
- **Operational Applications**:
  - **Grid Dispatch**: Day-ahead thermal generator scheduling and spinning reserve allocation.
  - **Renewable Integration**: Balancing variable wind/solar generation against aggregate demand.
  - **Wholesale Trading**: Minimizing imbalance penalties in day-ahead electricity spot markets.
- **Limitations**:
  - Weather features (temperature, wind chill) were not included in univariate series.
  - Multi-step forecasts degrade as errors compound over longer horizons ($h > 7$ days).

---

### Slide 10: Conclusion & References
- **Summary**:
  - Successfully built a complete, reproducible ML/DL electricity forecasting platform.
  - Demonstrated that Random Forest and Stacked LSTM achieve high predictive power ($R^2 > 0.83 - 0.91$).
  - Random Forest excels with explicit calendar features; LSTM excels at raw sequential learning.
- **References**:
  - [1] Hochreiter & Schmidhuber, "Long Short-Term Memory", *Neural Comp.*, 1997.
  - [2] L. Breiman, "Random Forests", *Machine Learning*, 2001.
  - [3] J. W. Taylor et al., "Univariate methods for electricity load forecasting", *IEEE Trans. Power Syst.*, 2006.
  - [4] Open Power System Data Platform, *OPSD Time Series*, 2020.
