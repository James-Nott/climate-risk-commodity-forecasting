"""
train_test.py

Author: James Nott
Date: 2025-08-11

Description:
    This script loads a time-series dataset, applies feature
    selection, and splits the data into training and testing sets
    using time-series cross-validation. It develops a XGBoost model which prints and
    writes out model performance.

Usage:
    python train_test.py

"""
import pandas as pd 
import numpy as np
import os
import xgboost as xgb
from james_nott_csc8099 import config
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# FIELDS 

CSV_PATH = config.DATASET_PATH      # Path to dataset
TARGET = "corn"
WRITE_OUT_ERR = False               # Error writeout status
CALC_PRICE_ERR = True               # whether actual price errors are printed
ABLATION_TEST = True                # whether to trim smdi features
HORIZON = 1                         # Prediction ahead
N_ESTIMATORS = 100                  # Number of trees
RANDOM_STATE = 42                   # Determinism
LEARNING_RATE = 0.01                # Slower learning set
MAX_DEPTH = 5                       # Depth of trees
N_SPLITS = 5                        # Number of splits for time series cross-validation
TEST_SIZE = 0.2                     # Proportion for final holdout test set
LOOKBACK_ALIGNMENT = 52             # Aligned with LSTM model so tests are comparative



# METHODS 

def add_lags(df, target) -> None:
    """Returns lagged features of target prices"""
    for lag in range(1, 5):
        if target == "corn":
            df[f'Corn_Last_lag{lag}'] = df['Corn_Last'].shift(lag) # Shift dataset by desired lag

        elif target == "soy":
            df[f'Soy_Last_lag{lag}'] = df['Soy_Last'].shift(lag)


def add_weekly_harmonics(df, peak_week, name, nharm=1):
    """Creates Fourier Terms with the desired number of harmonics"""
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
    
    # Common seasonality that applies to both corn and soy
    add_weekly_harmonics(df, peak_week=30, name='US')  # US harvest season
    
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

def ablation_test(df):
    """Drops SMDI columns for ablation test"""
    drop_list = ["extreme_decimal_united_states", "extreme_decimal_china", "extreme_decimal_brazil", "extreme_decimal_argentina"]
    df = df.drop(columns=drop_list)
    return df

def build_model(n_estimators = N_ESTIMATORS, random_state = RANDOM_STATE, learning_rate = LEARNING_RATE, max_depth = MAX_DEPTH):
    """Creation of model with parameters given in fields"""
    model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators= n_estimators,
    learning_rate = learning_rate,
    max_depth = max_depth,
    random_state= random_state)

    return model

def time_series_train_test_split(X, y, test_size=TEST_SIZE):
    """Split time series data maintaining temporal order"""
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def main(): 
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    # Use of Ablation test
    if ABLATION_TEST:
        df = ablation_test(df)
    else:
        # Add target-specific seasonality
        add_target_specific_seasonality(df, TARGET)

    # Add Lags 
    add_lags(df, TARGET)

    # Features 
    feature_cols = [c for c in df.columns]

    # Targets 
    target_raw = {"corn": "Corn_Last", "soy": "Soy_Last"}[TARGET]
    df[f"{TARGET}_log"] = np.log(df[target_raw])
    df["y"] = df[f"{TARGET}_log"].diff(periods=1).shift(-HORIZON) # For log price change 

    df = df.dropna(subset=feature_cols + ["y"]).copy()
    X_all = df[feature_cols].values
    y_all = df["y"].values.reshape(-1, 1)

    # Train and test 
    X_train, X_test, Y_train, Y_test = time_series_train_test_split(X_all, y_all)

    # Aligns with LSTM
    if LOOKBACK_ALIGNMENT and LOOKBACK_ALIGNMENT > 0:
        X_test = X_test[LOOKBACK_ALIGNMENT:]
        Y_test = Y_test[LOOKBACK_ALIGNMENT:]

    # Build, fit and predict
    model = build_model()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test) 

    # Naive baseline
    naive = np.zeros_like(Y_test)
    mae_naive = mean_absolute_error(Y_test, naive)

    # Metrics 
    mae = mean_absolute_error(Y_test, pred)


    if CALC_PRICE_ERR:
        # Price value MAE 
        P_prev_all = df[target_raw].values

        # Adjust for lookback alignment 
        k = LOOKBACK_ALIGNMENT
        start = len(X_train) + k
        end   = start + len(X_test)

        P_prev_train = P_prev_all[:len(X_train)]
        P_prev_test  = P_prev_all[start:end]    # Selects test data

        # Predicted and true next-step prices
        P_hat_test  = P_prev_test.flatten() * np.exp(pred.flatten())
        P_true_test = P_prev_test.flatten() * np.exp(Y_test.flatten())

        # Actual price metrics
        mae_price_test  = np.mean(np.abs(P_hat_test - P_true_test))
        P_hat_naive_test = P_prev_test.flatten()
        mae_price_naive = np.mean(np.abs(P_hat_naive_test - P_true_test))

        mape_price_model = np.mean(np.abs((P_true_test - P_hat_test) / P_true_test)) * 100.0
        mape_price_naive = np.mean(np.abs((P_true_test - P_hat_naive_test) / P_true_test)) * 100.0

    if WRITE_OUT_ERR:
        k = LOOKBACK_ALIGNMENT  # alignment used in LSTM for price conversion
        start_idx = len(X_train) + k
        end_idx   = start_idx + len(X_test)
        test_index = df.index[start_idx:end_idx]

        abs_err = np.abs(Y_test.flatten() - pred.flatten()) # Determines absolute error 
        out_df = pd.DataFrame({
            "date": test_index,
            "actual_log_return": Y_test.flatten(),
            "pred_log_return": pred.flatten(),
            "abs_error": abs_err
        }).sort_values("date")

        # File name encodes target and horizon
        csv_name = f"mae_log_return_xgboost_{TARGET}_h{HORIZON}.csv"
        csv_path = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", csv_name) # Writes to current directory
        out_df.to_csv(csv_path, index=False)
        print(f"[Saved] Per-timestep MAE (log return) for test set -> {csv_path}")
    
    # Prints metrics
    print(f"Price MAE (test): {mae_price_test:.6f}")
    print(f"Price MAE Naive (test): {mae_price_naive:.6f}")
    print(f"Price MAPE (model): {mape_price_model:.3f}%")
    print(f"Price MAPE (naive): {mape_price_naive:.3f}%")
    print(f"MAE: {mae}")
    print(f"MAE Naive: {mae_naive}")


    print("\n=== Time Series Cross-Validation ===")
    # Use only training data for cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    # Calculate cross-validation scores manually to see performance across folds
    cv_scores = []
    naive_scores = []
    fold = 1
    
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx] # Split data for fold
        y_cv_train, y_cv_val = Y_train[train_idx], Y_train[val_idx]
        
        model_cv = build_model()
        model_cv.fit(X_cv_train, y_cv_train)

        #Naive on folds
        naive_fold = np.zeros_like(y_cv_val)
        naive_mae = mean_absolute_error(y_cv_val, naive_fold)
        naive_scores.append(naive_mae)

        # Predicts for folds and adds to list 
        pred_cv = model_cv.predict(X_cv_val)
        mae_cv = mean_absolute_error(y_cv_val, pred_cv)
        cv_scores.append(mae_cv)
        
        print(f"Fold {fold}: MAE = {mae_cv:.6f}  Naive MAE = {naive_mae:.6f}")
        fold += 1
    
    # Results of Cross Validation
    print(f"Average CV MAE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores) * 2:.6f})")
    print(f"Average Naive MAE: {np.mean(naive_scores):.6f}")

    # Parameter tuning with time series cross-validation
    print("\n=== Grid Search with Time Series CV ===")
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Create time series cross-validator
    tscv_grid = TimeSeriesSplit(n_splits=N_SPLITS)
    
    # Grid search parameters
    grid_search = GridSearchCV(
        estimator=build_model(),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=tscv_grid,  # Use time series cross-validation
        verbose=1,
        n_jobs=-1
    )

    # Fit on training data only
    grid_search.fit(X_train, Y_train.ravel())
    
    print("Best params:", grid_search.best_params_)
    print("Best CV MAE:", -grid_search.best_score_)
    
    # Test best model on holdout test set
    best_model = grid_search.best_estimator_
    best_pred = best_model.predict(X_test)
    
    best_mae = mean_absolute_error(Y_test, best_pred)
    
    print(f"\n=== Best Model Test Set Performance ===")
    print(f"Test MAE: {best_mae}")


if __name__ == "__main__":
    main()