"""
train_test_split.py

Author: James Nott
Date: 2025-08-11

Description:
    This script loads a time-series dataset, applies feature
    scaling and selection, and splits the data into training and testing sets
    using time-series cross-validation. It develops a LSTM model which prints and
    writes out model performance.

Usage:
    python train_test_split.py

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
SEED = 42  # Fixed random seed to ensure reproducible results across runs
os.environ["PYTHONHASHSEED"] = str(SEED)  # Make Python's hashing repeatable
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # Request deterministic TensorFlow operations
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf                   
tf.random.set_seed(SEED)  # Set TensorFlow random seed for reproducibility                  
tf.config.experimental.enable_op_determinism()  # Ensure deterministic execution in TensorFlow
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input, GaussianNoise
from keras.models import Sequential
from keras import regularizers



CSV_PATH      = config.DATASET_PATH   # path to the dataset
TARGET        = "corn"
WRITE_OUT_ERR = False         # error write out control
CALC_PRICE_ERR = True         # whether actual price errors are printed 
ABLATION_TEST = True          # whether to trim smdi features
LOOK_BACK     = 52            # weeks of history fed to the network
HORIZON       = 1             # forecast horizon in weeks
UNITS         = 64            # LSTM units in first layer (second layer uses UNITS//2)
DROPOUT       = 0.2           # dropout rate
EPOCHS        = 300           # maximum training epochs
BATCH_SIZE    = 16            # mini‑batch size
RNN_INP_DROPOUT = 0.25          # hidden mask 
RNN_REC_DROPOUT = 0.15          # hidden to hidden mask
L2_COEFF        = 1e-4          # weight-decay strength
G_NOISE         = 0.01          # Noise (std) added



def build_lstm(input_shape: Tuple[int, int], units: int = UNITS, dropout: float = DROPOUT,
                rnn_inp_dropout: float = RNN_INP_DROPOUT, rnn_rec_dropout: float = RNN_REC_DROPOUT,
                l2_coef: float = L2_COEFF,
                gaussian_noise: float = G_NOISE,) -> Sequential:
    """Returns a 2 layer LSTM network using fields set."""
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
    """Returns lagged features of target prices"""
    for lag in range(1, 5):
        if target == "corn":
            df[f'Corn_Last_lag{lag}'] = df['Corn_Last'].shift(lag) # Shifts data by desired lag

        elif target == "soy":
            df[f'Soy_Last_lag{lag}'] = df['Soy_Last'].shift(lag)


def add_weekly_harmonics(df, peak_week, name, nharm=1):
    """Creates Fourier Terms with the desired number of harmonics"""
    wk = df.index.isocalendar().week.astype(int)

    # Map to 0-(period-1)   (period = 52.177 = 365.25 / 7)
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
        add_weekly_harmonics(df, peak_week=30, name='US')    # Harvest Season (Late July) 

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
        add_weekly_harmonics(df, peak_week=30, name='US')    # Harvest Season (Late July) 
        
        # Soy-specific interaction terms
        df['ext_united_states_sin'] = df['extreme_decimal_united_states'] * df['sin_US_1']
        df['ext_united_states_cos'] = df['extreme_decimal_united_states'] * df['cos_US_1']
        
        df['ext_china_cos_soy'] = df['extreme_decimal_china'] * df['cos_CH_S_1']
        
        df['ext_brazil_cos_soy'] = df['extreme_decimal_brazil'] * df['cos_BR_S_1']
        
        df['ext_argentina_cos'] = df['extreme_decimal_argentina'] * df['cos_ARG_1']

def ablation_test(df):
    """Drops SMDI columns for ablation test"""
    drop_list = ["extreme_decimal_united_states", "extreme_decimal_china", "extreme_decimal_brazil", "extreme_decimal_argentina"]
    df = df.drop(columns=drop_list)
    return df

def make_sequences(X: np.ndarray, y: np.ndarray, look_back: int, horizon = HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an array of sequences for model input"""
    seq_X, seq_y = [], []
    for i in range(len(X) - look_back - (horizon-1)): # Each sequence has lookback inputs and a horizon adjusted target
        seq_X.append(X[i : i + look_back])
        seq_y.append(y[i + look_back - 1])  # Pick the target exactly horizon after the last input
    return np.array(seq_X), np.array(seq_y)


# Main 

def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True) # Sets indexes
    df = df.set_index("date").sort_index()

    # Use of Ablation test
    if ABLATION_TEST:
        df = ablation_test(df)
    else:
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

    # 80/20 split
    split_idx = int(len(X_all) * 0.8)
    X_tr, X_te = X_all[:split_idx], X_all[split_idx:]
    y_tr, y_te = y_all[:split_idx], y_all[split_idx:]

    # Sequential feature selection
    selector = SequentialFeatureSelector(
        estimator=Lasso(alpha=0.01, max_iter=5000),
        n_features_to_select="auto",
        direction="forward", # Starts with a single feature
        cv=3
    ).fit(X_tr, y_tr.ravel())
    cols_sel = selector.get_support(indices=True)
    X_tr, X_te = X_tr[:, cols_sel], X_te[:, cols_sel] # Returns feautures for modelling

    # Scaling
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    X_te_s = scaler_X.transform(X_te)
    y_tr_s = scaler_y.fit_transform(y_tr)
    y_te_s = scaler_y.transform(y_te)

    # Build sequences
    X_train_seq, y_train_seq = make_sequences(X_tr_s, y_tr_s, LOOK_BACK)
    X_test_seq,  y_test_seq  = make_sequences(X_te_s, y_te_s, LOOK_BACK)

    # LSTM training
    model = build_lstm((LOOK_BACK, X_train_seq.shape[2]))
    es = EarlyStopping(patience=20, restore_best_weights=True, verbose=0) # Model stops when performance fails to increase
    model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0
    )
    
    # Inverse-scale and predictions for intepretation
    y_pred_s = model.predict(X_test_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s).ravel()
    y_true = scaler_y.inverse_transform(y_test_seq).ravel()

    if CALC_PRICE_ERR:
        # Previous prices before adjustments
        P_prev_all = df[target_raw].values

        # Split the base prices the same way as in model
        split_idx = int(len(X_all) * 0.8)
        P_prev_tr = P_prev_all[:split_idx]
        P_prev_te = P_prev_all[split_idx:]

        # Align base prices to the test sequences
        P_prev_test_seq = P_prev_te[LOOK_BACK - 1 : LOOK_BACK - 1 + len(y_true)]

        # Transform log returns to prices and compute MAE in price units
        P_hat_test  = P_prev_test_seq * np.exp(y_pred)
        P_true_test = P_prev_test_seq * np.exp(y_true)

        mae_price_model = np.mean(np.abs(P_hat_test - P_true_test))

        # Naive (zero-return) baseline in price units i.e. next price = current price
        mae_price_naive = np.mean(np.abs(P_prev_test_seq - P_true_test))

        mape_price_model = np.mean(np.abs((P_true_test - P_hat_test) / P_true_test)) * 100.0
        mape_price_naive = np.mean(np.abs((P_true_test - P_prev_test_seq) / P_true_test)) * 100.0

    # Return = 0 baseline
    naive_pred = np.zeros_like(y_true)  

    # Metrics
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_true, naive_pred)


    if WRITE_OUT_ERR: 
        idx_all = df.index  # Index after dropna and sorting
        test_start_in_all = split_idx + (LOOK_BACK - 1) # Finds start of test data
        test_end_in_all   = test_start_in_all + len(y_true) # Finds end
        test_dates = idx_all[test_start_in_all:test_end_in_all]

        abs_err = np.abs(y_true.flatten() - y_pred.flatten()) # Determines absolute error 
        out_df = pd.DataFrame({
            "date": test_dates,
            "actual_log_return": y_true.flatten(),
            "pred_log_return": y_pred.flatten(),
            "abs_error": abs_err
        }).sort_values("date")

        csv_name = f"mae_log_return_lstm_{TARGET}_h{HORIZON}.csv"
        csv_path = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", csv_name) # Path writes to current directory
        out_df.to_csv(csv_path, index=False)
        print(f"[Saved] Per-timestep MAE (log return) for test set -> {csv_path}")
    
    # Prints of metrics to terminal
    print(f"Price MAE (model): {mae_price_model:.6f}")
    print(f"Price MAE (naive): {mae_price_naive:.6f}")

    print(f"Price MAPE (model): {mape_price_model:.3f}%")
    print(f"Price MAPE (naive): {mape_price_naive:.3f}%")

    print(f"MAE model {mae_model:.4f} vs naive {mae_naive:.4f}")

    

if __name__ == "__main__":
    main()