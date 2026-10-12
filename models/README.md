# Saved Models and Artifacts

This directory stores trained model artifacts and preprocessing pipelines for reproducibility and production inference:

- `scaler.joblib`: Fitted `MinMaxScaler` on training set only (prevents data leakage).
- `best_lstm_model.keras`: Keras serialized HDF5/Native format LSTM neural network.
- `random_forest_model.joblib`: Serialized scikit-learn Random Forest Regressor.
- `model_metadata.json`: Hyperparameters, lookback window size (`n_steps`), feature names, and test metrics.
