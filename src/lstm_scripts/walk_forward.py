"""
walk_forward.py

Author: James Nott
Date: 2025-08-11

Description:
    This script loads a time-series dataset, applies feature
    selection, scaling and splits the data into folds for realistic model training/testing.
    It develops a LSTM model which prints and
    writes out model performance across folds and prints average as well.

Usage:
    python train_test.py

"""
from typing import Tuple
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lasso
import os, random, numpy as np
from james_nott_csc8099 import config
''' Sets deterministic lstm '''
SEED = 42                                 # Fixed random seed to ensure reproducible results across runs
os.environ["PYTHONHASHSEED"] = str(SEED)  # Make Python's hashing repeatable
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # Request deterministic TensorFlow operations
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf                   
tf.random.set_seed(SEED)    # Set TensorFlow random seed for reproducibility                
tf.config.experimental.enable_op_determinism() # Ensure deterministic execution in TensorFlow
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input, GaussianNoise
from keras.models import Sequential
from keras import regularizers

#Fields 

CSV_PATH      = config.DATASET_PATH   # path to the dataset
TARGET        = "corn"        
LOOK_BACK     = 52                    # weeks of history fed to the network
HORIZON       = 1                     # forecast horizon in weeks
N_SPLITS      = 5                     # number of walk‑forward folds
TEST_SIZE     = 104                   # test weeks per fold
UNITS         = 64                    # LSTM units in first layer (second layer uses UNITS//2)
DROPOUT       = 0.2                   # dropout rate
EPOCHS        = 300                   # maximum training epochs
BATCH_SIZE    = 16                    # mini‑batch size
RNN_INP_DROPOUT = 0.25                # hidden mask 
RNN_REC_DROPOUT = 0.15                # hidden to hidden mask
L2_COEFF        = 1e-4                # weight-decay strength
G_NOISE         = 0.01                # Noise (std) added



def build_lstm(input_shape: Tuple[int, int], units: int = UNITS, dropout: float = DROPOUT,
                rnn_inp_dropout: float = RNN_INP_DROPOUT, rnn_rec_dropout: float = RNN_REC_DROPOUT,
                l2_coef: float = L2_COEFF,
                gaussian_noise: float = G_NOISE,) -> Sequential:
    """Return a 2‑layer LSTM network."""
    l2 = regularizers.l2(l2_coef) # L2 on kernel

    model = Sequential([
        Input(shape=input_shape),
        GaussianNoise(gaussian_noise),

        # 1st Recurrent Block
        LSTM(units, return_sequences=True, 
             dropout=rnn_inp_dropout,
             recurrent_dropout=rnn_rec_dropout,
             kernel_regularizer=l2,
             recurrent_regularizer=l2
             ),
        Dropout(dropout),

        # 2nd Recurrent Block
        LSTM(units // 2,
             dropout=rnn_inp_dropout,
            recurrent_dropout=rnn_rec_dropout,
            kernel_regularizer=l2,
            recurrent_regularizer=l2
            ),
        Dropout(dropout),

        # Output head
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model

def add_lags(df, target) -> None:
    for lag in range(1, 5):
        if target == "corn":
            df[f'Corn_Last_lag{lag}'] = df['Corn_Last'].shift(lag) # Shifts data by desired lag

        elif target == "soy":
            df[f'Soy_Last_lag{lag}'] = df['Soy_Last'].shift(lag)


def add_weekly_harmonics(df, peak_week, name, nharm=1):
    wk = df.index.isocalendar().week.astype(int)

    # Map to 0-(period-1)   (period = 52.177 ≈ 365.25 / 7)
    period = 52.177
    angle_base = 2 * np.pi * ((wk - peak_week) % period) / period

    for h in range(1, nharm + 1):
        angle = h * angle_base
        df[f'sin_{name}_{h}'] = np.sin(angle)
        df[f'cos_{name}_{h}'] = np.cos(angle)

def add_target_specific_seasonality(df: pd.DataFrame, target: str) -> None:
    """Add seasonality features based on the target variable."""
    
    # Target-specific seasonality
    if target == "corn":
        # Corn-specific seasonality
        add_weekly_harmonics(df, peak_week=28, name='CH_C')  # China corn (Mid July)
        add_weekly_harmonics(df, peak_week=3, name='BR_C')   # Brazil corn (Late Jan)
        add_weekly_harmonics(df, peak_week=30, name='US') # Harvest Season (Late July) 

        # Corn-specific interaction terms
        df['ext_united_states_sin'] = df['extreme_decimal_united_states'] * df['sin_US_1']
        df['ext_united_states_cos'] = df['extreme_decimal_united_states'] * df['cos_US_1']
        
        df['ext_china_cos_corn'] = df['extreme_decimal_china'] * df['cos_CH_C_1']
        
        df['ext_brazil_cos_corn'] = df['extreme_decimal_brazil'] * df['cos_BR_C_1']

    elif target == "soy":
        # Soy-specific seasonality
        add_weekly_harmonics(df, peak_week=24, name='CH_S')  # China soy (Mid June)
        add_weekly_harmonics(df, peak_week=7, name='BR_S')   # Brazil soy (Late Feb)
        add_weekly_harmonics(df, peak_week=18, name='ARG')   # Argentina soy (April/May)
        add_weekly_harmonics(df, peak_week=30, name='US') # Harvest Season (Late July) 
        
        # Soy-specific interaction terms
        df['ext_united_states_sin'] = df['extreme_decimal_united_states'] * df['sin_US_1']
        df['ext_united_states_cos'] = df['extreme_decimal_united_states'] * df['cos_US_1']
        
        df['ext_china_cos_soy'] = df['extreme_decimal_china'] * df['cos_CH_S_1']
        
        df['ext_brazil_cos_soy'] = df['extreme_decimal_brazil'] * df['cos_BR_S_1']
        
        df['ext_argentina_cos'] = df['extreme_decimal_argentina'] * df['cos_ARG_1']


def make_sequences(X: np.ndarray, y: np.ndarray, look_back: int, horizon = HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    seq_X, seq_y = [], []
    for i in range(len(X) - look_back - (horizon-1)): # Each sequence has lookback inputs and a horizon adjusted target
        seq_X.append(X[i : i + look_back])
        seq_y.append(y[i + look_back - 1]) # Pick the target exactly horizon after the last input 
    return np.array(seq_X), np.array(seq_y)


# Main 

def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    # Add target-specific seasonality
    add_target_specific_seasonality(df, TARGET)
    # Add Lags 
    add_lags(df, TARGET)
     

    # Target engineering
    target_raw = {"corn": "Corn_Last", "soy": "Soy_Last"}[TARGET]
    df[f"{TARGET}_log"] = np.log(df[target_raw])
    df["y"] = df[f"{TARGET}_log"].diff(periods=1).shift(-HORIZON) # For log price change 

     # Feature selection
    feature_cols = [c for c in df.columns if c not in set(["y"])]

    df = df.dropna(subset=feature_cols + ["y"]).copy()
    X_all = df[feature_cols].values
    y_all = df["y"].values.reshape(-1, 1) # Target

    # Base prices aligned with X_all / y_all AFTER dropna
    P_prev_all = df[target_raw].values
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_results = []


    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), 1):
        print(f"\nFold {fold}/{N_SPLITS} – train up to {df.index[train_idx[-1]].date()}, test {len(test_idx)} weeks")

        X_tr, X_te = X_all[train_idx], X_all[test_idx] # Portions for training and testing
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # Using a Sequential Feature Selector on each fold
        selector = SequentialFeatureSelector(
            estimator=Lasso(alpha=0.01, max_iter=5000),
            n_features_to_select="auto",
            direction="forward", # Starts with one feature
            cv=3
         ).fit(X_tr, y_tr.ravel())

        cols_sel = selector.get_support(indices=True)
        X_tr, X_te = X_tr[:, cols_sel], X_te[:, cols_sel]


         # Check if we have enough training data for sequences
        if len(train_idx) <= LOOK_BACK:
            print(f"Skipping fold {fold}: insufficient training data ({len(train_idx)} <= {LOOK_BACK})")
            continue
        
        # Scaling for inputs
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_tr)
        y_train_scaled = scaler_y.fit_transform(y_tr)
        X_test_scaled  = scaler_X.transform(X_te)
        y_test_scaled  = scaler_y.transform(y_te)

        X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, LOOK_BACK) # Sequences for input data
        X_test_seq,  y_test_seq  = make_sequences(X_test_scaled, y_test_scaled, LOOK_BACK)

        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2]) # Shape for inputs
        model = build_lstm(input_shape)
        es = EarlyStopping(patience=20, restore_best_weights=True, verbose=0)

        model.fit(X_train_seq, y_train_seq, validation_split=0.15, epochs=EPOCHS, # Fitting fold
                  batch_size=BATCH_SIZE, verbose=0, callbacks=[es])

        y_pred_scaled = model.predict(X_test_seq, verbose=0) # Prediction for fold

        # Inverse transform predicted values
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = y_pred.ravel()

        # Inverse transform the true target values
        y_true = scaler_y.inverse_transform(y_test_seq)
        y_true = y_true.ravel()

        # Actual price metrics for each fold
        # Align base prices for this fold’s test indices
        P_prev_te = P_prev_all[test_idx] 

        # Aligns with make sequences
        P_prev_test_seq = P_prev_te[LOOK_BACK - 1 : LOOK_BACK - 1 + len(y_true)]

        # Model price forecasts and ground truth prices
        P_hat = P_prev_test_seq * np.exp(y_pred)
        P_true = P_prev_test_seq * np.exp(y_true)

        # MAE in price units
        mae_price_model = np.mean(np.abs(P_hat - P_true))
        # Naive (zero‑return) baseline: next price equals current price
        mae_price_naive = np.mean(np.abs(P_prev_test_seq - P_true))
        mape_price_model = np.nanmean(np.abs((P_true - P_hat) / P_true)) * 100.0
        mape_price_naive = np.nanmean(np.abs((P_true - P_prev_test_seq) / P_true)) * 100.0

        naive_pred = np.zeros_like(y_true)  # return = 0 baseline

        # Calculate Performance Metrics (log returns)
        mae_model = mean_absolute_error(y_true, y_pred)
        mae_naive = mean_absolute_error(y_true, naive_pred)
       
        # Results added to list of all folds
        fold_results.append({"fold": fold, "mae_model": mae_model, "mae_naive": mae_naive, "mae_price_model": mae_price_model, "mae_price_naive": mae_price_naive,
    "mape_price_model": mape_price_model, "mape_price_naive": mape_price_naive,})
        
        # Print individual fold results
        print(f"MAE model {mae_model:.4f} vs naive {mae_naive:.4f}")
        print(f"Price MAE (model): {mae_price_model:.6f} | (naive): {mae_price_naive:.6f}")
        print(f"Price MAPE (model): {mape_price_model:.3f}% | (naive): {mape_price_naive:.3f}%")
        
    # Print results across all folds and averages of results
    res_df = pd.DataFrame(fold_results)
    print("\nWalk forward summary")
    print(res_df)
    print("Avg MAE model :", res_df["mae_model"].mean())
    print("Avg MAE naive :", res_df["mae_naive"].mean())
    print("Avg Price MAE model :", res_df["mae_price_model"].mean())
    print("Avg Price MAE naive :", res_df["mae_price_naive"].mean())
    print("Avg Price MAPE model :", res_df["mape_price_model"].mean())
    print("Avg Price MAPE naive :", res_df["mape_price_naive"].mean())


if __name__ == "__main__":
    main()