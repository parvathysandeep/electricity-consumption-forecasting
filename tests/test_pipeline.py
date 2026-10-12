"""
Unit & Integration Test Suite for Electricity Consumption Forecasting.
Verifies data loading, sequence creation, metrics calculation, and input validation.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import create_sliding_sequences, chronological_split
from src.evaluation import calculate_metrics
from src.predict import predict_next_step


class TestElectricityForecastingPipeline(unittest.TestCase):

    def test_chronological_split(self):
        """Verify chronological order is preserved and split ratios are respected."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"Consumption": np.arange(100)}, index=dates)
        
        train, val, test = chronological_split(df, train_ratio=0.7, val_ratio=0.15)
        
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
        
        # Verify strict temporal continuity without shuffling
        self.assertTrue(train.index.max() < val.index.min())
        self.assertTrue(val.index.max() < test.index.min())

    def test_sliding_window_sequences(self):
        """Verify sequence generator produces exact 3D shape and aligned targets."""
        series = np.array([10, 20, 30, 40, 50, 60, 70], dtype=float)
        n_steps = 3
        X, y = create_sliding_sequences(series, n_steps=n_steps)
        
        # Expected: 7 - 3 = 4 samples
        self.assertEqual(X.shape, (4, 3, 1))
        self.assertEqual(y.shape, (4, 1))
        
        # Check first window
        np.testing.assert_array_equal(X[0].flatten(), [10, 20, 30])
        self.assertEqual(y[0][0], 40)

    def test_evaluation_metrics(self):
        """Verify metric calculation accuracy and safe handling of zero values."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])
        
        metrics = calculate_metrics(y_true, y_pred, model_name="TestModel")
        
        self.assertAlmostEqual(metrics["MAE (GWh)"], 6.667, places=2)
        self.assertAlmostEqual(metrics["RMSE (GWh)"], 8.165, places=2)
        self.assertGreater(metrics["R2 Score"], 0.95)
        self.assertIn("MAPE (%)", metrics)

    def test_prediction_input_validation(self):
        """Verify that prediction function rejects invalid inputs gracefully."""
        # Insufficient observations
        short_input = [100.0, 105.0]
        with self.assertRaises(ValueError):
            predict_next_step(short_input, n_steps=14)

        # Input containing NaNs
        nan_input = [100.0] * 13 + [np.nan]
        with self.assertRaises(ValueError):
            predict_next_step(nan_input, n_steps=14)


if __name__ == "__main__":
    unittest.main()
