# Viva Exam Preparation: Electricity Demand Forecasting
## Comprehensive Questions & Model Answers for Project Defense

---

### Q1: Why was electricity consumption forecasting selected as the project topic?
**Answer**: Electricity cannot be efficiently stored in bulk across national transmission grids; generation must continuously match instantaneous demand. Inaccurate forecasts cause expensive spinning reserve activation, frequency drops, or blackouts. Accurately predicting electricity consumption is a high-impact, real-world regression problem with strong multi-scale seasonality (daily, weekly, and annual cycles), making it an ideal testbed to compare traditional machine learning against deep recurrent architectures.

---

### Q2: Why is LSTM particularly well suited for time-series forecasting compared to standard RNNs?
**Answer**: Standard Recurrent Neural Networks (RNNs) suffer from the **vanishing and exploding gradient problem**. As sequence lengths grow, backpropagation through time repeatedly multiplies weight matrices, causing gradients to diminish exponentially to zero. LSTMs solve this by incorporating a dedicated **Cell State ($C_t$)** that acts as an internal highway. Information flows linearly along the cell state, regulated by multiplicative gates that control when to remember, update, or forget information over long lookback windows.

---

### Q3: How do the LSTM gates function mathematically?
**Answer**: An LSTM unit operates via three primary gates:
1. **Forget Gate ($f_t$)**: Determines what proportion of old cell state to discard using a sigmoid activation:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \in [0, 1]$$
2. **Input Gate ($i_t$) & Candidate ($\tilde{C}_t$)**: Determines what new information to store:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i), \quad \tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$)**: Combines forgotten past and gated candidate input:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t$)**: Controls which parts of the cell state form the hidden state:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$

---

### Q4: Why are baseline models necessary in machine learning projects?
**Answer**: Baseline models establish the minimum acceptable performance threshold. A complex deep learning model is useless if it cannot outperform a trivial heuristic (such as naive persistence, $\hat{y}_{t+1} = y_t$). Implementing baseline models (Persistence, Linear Regression, Random Forest) provides a rigorous benchmark to quantify whether neural network complexity delivers genuine empirical gains.

---

### Q5: Why can't we use standard `train_test_split(shuffle=True)` for time-series data?
**Answer**: Standard random shuffling destroys the chronological order of observations and causes **lookahead bias (data leakage)**. If tomorrow's demand is included in the training set to predict today's demand, the model achieves falsely optimistic training metrics but fails completely in real production. Time-series data must always be partitioned **chronologically**: the past is used for training, the intermediate period for validation, and the future for testing.

---

### Q6: What is data leakage and how did your pipeline prevent it?
**Answer**: Data leakage occurs when information from outside the training partition leaks into model fitting. We prevented leakage through three strict rules:
1. **Chronological Splitting**: 70% Train (2006–2014), 15% Validation (2014–2016), 15% Test (2016–2017).
2. **Isolated Normalization**: The `MinMaxScaler` was fit **only** on the training set. Test and validation data were transformed using the pre-computed training parameters.
3. **Shifted Windowing**: All rolling window statistics were computed using `.shift(1)`, ensuring the current observation $y_t$ is never used to build features for day $t$.

---

### Q7: Why is feature scaling required for LSTM networks but not strictly for Random Forests?
**Answer**: LSTMs use activation functions like $\tanh$ and $\text{sigmoid}$ which saturate when inputs are large, causing gradients to vanish. Furthermore, neural networks optimize weights via gradient descent, which converges much faster when features are normalized to a consistent range $[0, 1]$. In contrast, Random Forests make axis-aligned split decisions based on feature order/thresholds, making them invariant to monotonic scaling.

---

### Q8: What do MAE, RMSE, and MAPE measure, and when do they differ?
**Answer**:
- **Mean Absolute Error (MAE)**: Measures average absolute magnitude of errors in original physical units (GWh). It treats all errors linearly.
- **Root Mean Squared Error (RMSE)**: Squares errors before averaging and takes the square root. Because errors are squared, RMSE penalizes large outlier errors much more severely than MAE. A large gap between RMSE and MAE indicates high error variance or occasional large forecast misses.
- **Mean Absolute Percentage Error (MAPE)**: Expresses error as a percentage relative to the actual value, providing an intuitive, scale-independent metric for business stakeholders.

---

### Q9: How was the LSTM model trained and how did you prevent overfitting?
**Answer**: The model was trained using the **Adam optimizer** with Mean Squared Error loss. To prevent overfitting:
1. **Dropout Layers ($p=0.2$)**: Randomly zeroed 20% of activations during training to prevent co-adaptation of neurons.
2. **Validation Set Monitoring**: Performance was tracked on an independent 15% validation partition.
3. **Early Stopping with Restored Weights**: Training halted if validation loss did not improve for 8 consecutive epochs, restoring the best model weights.
4. **Learning Rate Reduction**: `ReduceLROnPlateau` halved the learning rate when validation loss plateaued.

---

### Q10: How does the model generate a prediction from new inputs?
**Answer**: Given the past 14 days of electricity consumption:
1. Input validation confirms shape ($N \ge 14$), numerical validity, and absence of NaNs.
2. The sequence is normalized using the saved training `scaler.joblib`.
3. The scaled array is reshaped to 3D tensor format $(1, 14, 1)$.
4. The trained Keras LSTM executes a forward pass: $\hat{y}_{\text{scaled}} = \text{model}(X)$.
5. The output is inverse-transformed back to physical units (GWh):
   $$\hat{y} = \hat{y}_{\text{scaled}} \cdot (y_{\max} - y_{\min}) + y_{\min}$$

---

### Q11: What do your actual project results demonstrate?
**Answer**: On the 656-day held-out test set:
- **Random Forest**: $MAE = 24.94\text{ GWh}$, $RMSE = 45.94\text{ GWh}$, $MAPE = 1.91\%$, $R^2 = 0.9178$
- **Linear Regression**: $MAE = 34.05\text{ GWh}$, $RMSE = 55.96\text{ GWh}$, $MAPE = 2.58\%$, $R^2 = 0.8781$
- **Stacked LSTM**: $MAE = 37.96\text{ GWh}$, $RMSE = 64.31\text{ GWh}$, $MAPE = 2.91\%$, $R^2 = 0.8390$
- **Persistence Baseline**: $MAE = 102.04\text{ GWh}$, $RMSE = 149.47\text{ GWh}$, $MAPE = 7.76\%$, $R^2 = 0.1300$

All learning models cut naive error by over 60%, verifying that Machine Learning and Deep Learning effectively capture electricity consumption patterns.

---

### Q12: Why did Random Forest achieve slightly lower error than LSTM in this experiment?
**Answer**:
1. **Explicit Weekly Calendar Features**: Electricity consumption on a daily scale has an exceptionally rigid 7-day periodicity (Monday–Friday high, Saturday/Sunday low). Providing explicit `day_of_week` and `is_weekend` columns allows decision trees to isolate weekend demand with exact step-function splits.
2. **Dataset Size for Deep Learning**: 4,383 daily records (~3,000 training days) is relatively modest for deep recurrent networks, which typically shine on massive high-frequency datasets (e.g., 15-minute or hourly resolution over tens of thousands of timesteps).
3. **Unassisted Sequential Learning in LSTM**: The LSTM achieved $2.91\%$ MAPE **purely from raw sequential lookback** without hand-crafted calendar features, proving its powerful internal feature extraction ability.

---

### Q13: How does multi-step autoregressive forecasting work, and what is its main drawback?
**Answer**: In iterative autoregressive rollout, the model predicts day $t+1$. This predicted value is then appended to the input sequence as a pseudo-observation to forecast day $t+2$, repeating until horizon $h$. The main drawback is **error accumulation**: any small error in early steps is fed back into the model, compounding into higher uncertainty over longer horizons ($h > 7$ days).

---

### Q14: What are the primary limitations of your experiment?
**Answer**:
1. **Univariate Setting**: We used historical load only. Electricity demand is strongly influenced by weather (temperature, cooling degree days, wind chill).
2. **Daily Aggregation**: Real-time grid balancing requires sub-hourly or hourly forecasts.
3. **Point Forecasts Only**: The model outputs single point estimates rather than probabilistic prediction intervals (e.g., 90% confidence bands).

---

### Q15: How could this system be improved for industrial grid deployment?
**Answer**:
1. **Exogenous Weather Variables**: Ingest Numerical Weather Prediction (NWP) forecasts for temperature and solar irradiance.
2. **Hierarchical / Substation Level**: Expand from national aggregate load down to regional transmission nodes.
3. **Modern Architectures**: Benchmark Temporal Fusion Transformers (TFT) or N-BEATS, which combine deep sequence modeling with interpretable attention weights.
4. **Probabilistic Loss**: Train with Pinball / Quantile loss to output $P_{10}, P_{50}, P_{90}$ forecast intervals for risk-aware power dispatch.
