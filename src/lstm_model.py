"""
LSTM Neural Network Architecture and Training Module.
Implements Long Short-Term Memory recurrent networks for electricity demand forecasting.

Mathematical Formulation of LSTM Gates:
1. Forget Gate:    f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
2. Input Gate:     i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
3. Candidate Cell: C~_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
4. Cell Update:    C_t = f_t * C_{t-1} + i_t * C~_t
5. Output Gate:    o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
6. Hidden State:   h_t = o_t * tanh(C_t)
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np


DEFAULT_LSTM_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_lstm_model.keras")


def build_lstm_model(
    input_shape: tuple,
    lstm_units: list = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Constructs a stacked LSTM regression model for time-series forecasting.
    
    Args:
        input_shape: Tuple of (time_steps, features), e.g. (14, 1)
        lstm_units: List of hidden units per LSTM layer.
        dropout_rate: Dropout fraction for regularization.
        learning_rate: Initial Adam learning rate.
    """
    model = Sequential(name="Electricity_Forecasting_LSTM")
    model.add(Input(shape=input_shape))
    
    # Layer 1
    if len(lstm_units) > 1:
        model.add(LSTM(units=lstm_units[0], return_sequences=True))
        model.add(Dropout(dropout_rate))
        # Layer 2
        model.add(LSTM(units=lstm_units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(units=lstm_units[0], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
    # Fully connected layers
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=1, activation="linear")) # Single-step continuous demand output

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    
    print("[LSTM] Model Architecture Summary:")
    model.summary()
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_save_path: str = DEFAULT_LSTM_PATH,
    lstm_units: list = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10
):
    """
    Trains the LSTM network with early stopping and learning rate scheduling.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"[LSTM] Starting training for up to {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"[LSTM] Training complete. Best model saved to: {model_save_path}")
    return model, history


def load_trained_lstm(model_path: str = DEFAULT_LSTM_PATH) -> Sequential:
    """
    Loads saved Keras LSTM model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Train the model first.")
    return load_model(model_path)
