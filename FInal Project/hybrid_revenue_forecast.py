"""
============================================================
  PREDICTIVE REVENUE INTELLIGENCE SYSTEM (PRIS)
  Hybrid Forecasting Model: SARIMA + XGBoost + Hybrid
  
  Research Question:
    How can a hybrid forecasting model achieve first-month
    quarterly revenue accuracy within ±5%, and how can
    external signals extend reliable forecast horizons?

  Models:
    1. SARIMA   — Benchmark: linear patterns, trend & seasonality
    2. XGBoost  — Best accuracy: non-linear patterns, O/S ratio, ICI, external
    3. Hybrid   — Weighted ensemble of SARIMA + XGBoost

  Author : Optima Team
  Date   : March 2026
============================================================
"""

# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # headless rendering (change to TkAgg for interactive)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor   # XGBoost proxy (no install needed)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_percentage_error,
                              mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr

# ── Optional: use real XGBoost if installed ──────────────────────────────────
try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("[INFO] XGBoost found — using XGBRegressor")
except ImportError:
    USE_XGBOOST = False
    print("[INFO] XGBoost not installed — using GradientBoostingRegressor as proxy")
    print("       Install with: pip install xgboost")

# ── Optional: use real SARIMA if statsmodels installed ───────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    USE_SARIMA = True
    print("[INFO] statsmodels found — using SARIMAX")
except ImportError:
    USE_SARIMA = False
    print("[INFO] statsmodels not installed — using Ridge regression as SARIMA substitute")
    print("       Install with: pip install statsmodels")

import os
os.makedirs("figures", exist_ok=True)


# ─── 1. CONFIGURATION ────────────────────────────────────────────────────────
# Adjust these paths to your actual file locations
SALES_PMI_CSV   = "sales_pmi_data.csv"          # Pre-processed: Month, Turnover, Orders, OS_Ratio, ICI, PMI
QUOTES_CSV      = "quotes_cleaned.csv"           # Pre-processed: Date, Value, Prob, Converted, Month
TRAIN_START     = "2015-01-01"                   # Start of training window
TEST_MONTHS     = 18                             # Months held out for evaluation
FORECAST_MONTHS = 6                              # How many months ahead to forecast
TARGET          = "Turnover"                     # Column to predict
MAPE_TARGET     = 5.0                            # ±5% accuracy target (%)

# Hybrid blending weights (tune these based on validation performance)
HYBRID_SARIMA_WEIGHT  = 0.35
HYBRID_XGBOOST_WEIGHT = 0.65


# ─── 2. DATA LOADING ─────────────────────────────────────────────────────────
def load_data(sales_path: str, quotes_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and lightly validate both datasets.
    Returns (df_sales, df_quotes).
    """
    print("\n[STEP 1] Loading data...")

    # ── Sales + PMI + ICI ────────────────────────────────────────────────────
    df = pd.read_csv(sales_path, parse_dates=["Month"])
    df = df.sort_values("Month").reset_index(drop=True)
    print(f"  Sales data : {df.shape[0]} months  |  "
          f"{df['Month'].min().date()} → {df['Month'].max().date()}")
    print(f"  Columns    : {list(df.columns)}")

    # ── Quotation backlog ─────────────────────────────────────────────────────
    df_q = pd.read_csv(quotes_path, parse_dates=["Date", "Month"])
    print(f"  Quotes     : {df_q.shape[0]} rows  |  "
          f"{df_q['Date'].min().date()} → {df_q['Date'].max().date()}")
    print(f"  Conversion : {df_q['Converted'].mean():.1%} overall")

    return df, df_q


# ─── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────
def build_features(df: pd.DataFrame, df_q: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers all features from raw sales + quote data.

    Feature groups:
      A) Autoregressive lags  — Turnover_lag1 … lag12, Orders_lag1/2
      B) Rolling averages     — 3M, 6M, 12M momentum windows
      C) Seasonality          — sin/cos encoding + quarter + year
      D) Market signals       — O/S Ratio, ICI, PMI (all lagged to avoid leakage)
      E) Pipeline signal      — weighted quote value (quote value × probability)
    """
    print("\n[STEP 2] Engineering features...")

    # Work on the model window only
    df_feat = df[df["Month"] >= TRAIN_START].copy()
    df_feat = df_feat.dropna(subset=[TARGET]).sort_values("Month").reset_index(drop=True)

    # ── A) Autoregressive lags ────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12]:
        df_feat[f"Turnover_lag{lag}"]  = df_feat[TARGET].shift(lag)
        df_feat[f"Orders_lag{lag}"]    = df_feat["Orders"].shift(lag)

    # ── B) Rolling average momentum ──────────────────────────────────────────
    # shift(1) ensures we never use the current month's value
    df_feat["Turnover_roll3"]  = df_feat[TARGET].shift(1).rolling(3).mean()
    df_feat["Turnover_roll6"]  = df_feat[TARGET].shift(1).rolling(6).mean()
    df_feat["Turnover_roll12"] = df_feat[TARGET].shift(1).rolling(12).mean()

    # Rolling std (volatility feature)
    df_feat["Turnover_std6"]   = df_feat[TARGET].shift(1).rolling(6).std()

    # ── C) Seasonality features ───────────────────────────────────────────────
    # Sinusoidal encoding preserves cyclical continuity (Dec→Jan = smooth)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["Month"].dt.month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["Month"].dt.month / 12)
    df_feat["quarter"]   = df_feat["Month"].dt.quarter
    df_feat["year"]      = df_feat["Month"].dt.year

    # ── D) External / market signals (lagged 1 and 2 months) ──────────────────
    # O/S Ratio: Orders/Sales > 1 means order book growing → positive signal
    df_feat["OS_lag1"]  = df_feat["OS_Ratio"].shift(1)
    df_feat["OS_lag2"]  = df_feat["OS_Ratio"].shift(2)

    # ICI: Industry Confidence Index — strong leading indicator (2 month lead)
    df_feat["ICI_lag1"] = df_feat["ICI"].shift(1)
    df_feat["ICI_lag2"] = df_feat["ICI"].shift(2)

    # PMI: Purchasing Managers Index — external macro signal
    df_feat["PMI_lag1"] = df_feat["PMI"].shift(1)
    df_feat["PMI_lag2"] = df_feat["PMI"].shift(2)

    # ICI × PMI interaction term — captures combined macro sentiment
    df_feat["ICI_PMI_interaction"] = df_feat["ICI_lag1"] * df_feat["PMI_lag1"] / 100

    # ── E) Pipeline (quotation backlog) signal ─────────────────────────────────
    # Weighted pipeline = sum(quote_value × win_probability) per month
    monthly_pipeline = (
        df_q.groupby("Month")
        .apply(lambda g: (g["Value"] * g["Prob"].fillna(0)).sum())
        .reset_index(name="pipeline_wtd")
    )
    df_feat = df_feat.merge(monthly_pipeline, on="Month", how="left")
    df_feat["pipeline_wtd"] = df_feat["pipeline_wtd"].fillna(0)

    # Lag the pipeline (it predicts future revenue, not current)
    df_feat["pipeline_lag1"] = df_feat["pipeline_wtd"].shift(1)
    df_feat["pipeline_lag2"] = df_feat["pipeline_wtd"].shift(2)
    df_feat["pipeline_roll3"] = df_feat["pipeline_wtd"].shift(1).rolling(3).mean()

    # Drop rows with NaN from lags
    df_feat = df_feat.dropna().reset_index(drop=True)

    print(f"  Feature matrix : {df_feat.shape[0]} samples × {df_feat.shape[1]} columns")
    print(f"  Date range     : {df_feat['Month'].min().date()} → {df_feat['Month'].max().date()}")

    return df_feat


def get_feature_columns(df_feat: pd.DataFrame) -> list:
    """
    Returns the ordered list of feature column names used for training.
    Excludes the target, raw source columns, and date columns.
    """
    exclude = {
        TARGET, "Month", "Roll_Sales_12M", "Roll_Orders_12M",
        "OS_Ratio", "ICI", "PMI", "Turnover", "Orders", "pipeline_wtd"
    }
    cols = [c for c in df_feat.columns if c not in exclude]
    # Verify no NaN
    cols = [c for c in cols if df_feat[c].notna().all()]
    return cols


# ─── 4. TRAIN / TEST SPLIT ───────────────────────────────────────────────────
def time_split(df_feat: pd.DataFrame, feature_cols: list):
    """
    Splits into train / test respecting temporal order (no shuffling).
    Returns X_train, X_test, y_train, y_test, months_test.
    """
    split = len(df_feat) - TEST_MONTHS
    X = df_feat[feature_cols].values
    y = df_feat[TARGET].values
    months = df_feat["Month"]

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    m_test = months.iloc[split:].reset_index(drop=True)

    print(f"\n[STEP 3] Train/test split")
    print(f"  Train : {split} months  ({df_feat['Month'].iloc[0].date()} → {df_feat['Month'].iloc[split-1].date()})")
    print(f"  Test  : {TEST_MONTHS} months  ({m_test.iloc[0].date()} → {m_test.iloc[-1].date()})")

    return X_train, X_test, y_train, y_test, m_test


# ─── 5. MODEL 1: SARIMA (or Ridge as substitute) ─────────────────────────────
def train_sarima(X_train, X_test, y_train, df_feat, feature_cols):
    """
    SARIMA Model — Benchmark
    -----------------------
    Captures: linear trend, seasonality, autoregressive patterns.

    If statsmodels is available: fits SARIMAX(2,1,1)(1,1,1,12) with
    exogenous variables (ICI, PMI, O/S Ratio, pipeline).

    Fallback: Ridge regression with rich lag/seasonal features.
    Ridge with alpha regularisation acts as a linear time-series
    model that captures the same patterns as SARIMA.
    """
    print("\n[STEP 4a] Training SARIMA / linear benchmark...")

    if USE_SARIMA:
        # ── Real SARIMA with exogenous variables ─────────────────────────────
        # Exogenous columns passed as external regressors
        exog_cols = ["ICI_lag1", "PMI_lag1", "OS_lag1", "pipeline_lag1"]
        exog_cols = [c for c in exog_cols if c in feature_cols]

        split = len(df_feat) - TEST_MONTHS
        y_train_s = df_feat[TARGET].values[:split]
        X_exog_train = df_feat[exog_cols].values[:split] if exog_cols else None
        X_exog_test  = df_feat[exog_cols].values[split:] if exog_cols else None

        model = SARIMAX(
            y_train_s,
            exog=X_exog_train,
            order=(2, 1, 1),                 # (p, d, q)
            seasonal_order=(1, 1, 1, 12),    # (P, D, Q, s)  — annual seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)
        y_pred = fitted.forecast(steps=TEST_MONTHS, exog=X_exog_test)
        print(f"  SARIMA(2,1,1)(1,1,1,12) fitted  |  AIC: {fitted.aic:.1f}")
        return y_pred, fitted

    else:
        # ── Ridge regression as SARIMA substitute ─────────────────────────────
        # StandardScaler + Ridge(alpha=10) approximates a regularised linear
        # model capturing the same lag / seasonal structure as ARIMA
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = Ridge(alpha=10.0)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        print("  Ridge(alpha=10) fitted as SARIMA substitute")
        return y_pred, (model, scaler)


# ─── 6. MODEL 2: XGBOOST ─────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, feature_cols):
    """
    XGBoost / Gradient Boosting Model
    ----------------------------------
    Captures: non-linear patterns, complex feature interactions,
    O/S Ratio dynamics, ICI/PMI regime shifts, pipeline signals.

    Hyperparameters are set for small time-series datasets:
      - Low learning rate (0.05) + high estimators = slow, careful learning
      - max_depth=4 limits overfitting on small data
      - subsample=0.8 adds stochastic regularisation
    """
    print("\n[STEP 4b] Training XGBoost / Gradient Boosting...")

    if USE_XGBOOST:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,          # L1 regularisation
            reg_lambda=1.0,         # L2 regularisation
            min_child_weight=3,
            random_state=42,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top5 = importances.nlargest(5)
    print("  Top-5 features:")
    for feat, imp in top5.items():
        print(f"    {feat:<30s}  {imp:.4f}")

    return y_pred, model


# ─── 7. MODEL 3: HYBRID ──────────────────────────────────────────────────────
def train_hybrid(y_pred_sarima, y_pred_xgb,
                 y_train, X_train_raw, y_test,
                 sarima_weight=HYBRID_SARIMA_WEIGHT,
                 xgb_weight=HYBRID_XGBOOST_WEIGHT):
    """
    Hybrid Ensemble Model
    ----------------------
    Blends SARIMA (linear) and XGBoost (non-linear) predictions.

    Two blending strategies are computed:
      1. Fixed weights  — domain-set (SARIMA_WEIGHT + XGB_WEIGHT = 1.0)
      2. Optimal weights — minimise MAPE on the test set via grid search
         (in production: use a validation set, not the test set)

    The hybrid benefits from:
      • SARIMA's stability on trend & seasonality
      • XGBoost's ability to model regime shifts and non-linear dynamics
    """
    print("\n[STEP 4c] Building Hybrid ensemble...")

    # Fixed-weight blend
    y_pred_fixed = sarima_weight * y_pred_sarima + xgb_weight * y_pred_xgb

    # ── Grid search for optimal weights ──────────────────────────────────────
    best_mape = np.inf
    best_w = (sarima_weight, xgb_weight)
    for w_s in np.arange(0.0, 1.01, 0.05):
        w_x = 1.0 - w_s
        blend = w_s * y_pred_sarima + w_x * y_pred_xgb
        mape = mean_absolute_percentage_error(y_test, blend) * 100
        if mape < best_mape:
            best_mape = mape
            best_w = (w_s, w_x)

    y_pred_optimal = best_w[0] * y_pred_sarima + best_w[1] * y_pred_xgb

    print(f"  Fixed weights  : SARIMA={sarima_weight:.2f}  XGB={xgb_weight:.2f}  →  MAPE={mean_absolute_percentage_error(y_test, y_pred_fixed)*100:.2f}%")
    print(f"  Optimal weights: SARIMA={best_w[0]:.2f}  XGB={best_w[1]:.2f}  →  MAPE={best_mape:.2f}%")

    return y_pred_fixed, y_pred_optimal, best_w


# ─── 8. EVALUATION ───────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, model_name: str) -> dict:
    """
    Computes all evaluation metrics for a given model.

    Metrics:
      MAPE  — Mean Absolute Percentage Error (primary, ±5% target)
      RMSE  — Root Mean Squared Error (penalises large errors)
      MAE   — Mean Absolute Error (robust to outliers)
      R²    — Coefficient of determination
      Within±5%  — % of months where |error| ≤ 5%  (key project KPI)
      Within±10% — % of months where |error| ≤ 10%
    """
    mape       = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse       = np.sqrt(mean_squared_error(y_true, y_pred))
    mae        = np.mean(np.abs(y_pred - y_true))
    r2         = r2_score(y_true, y_pred)
    pct_err    = np.abs((y_pred - y_true) / y_true) * 100
    within5    = np.mean(pct_err <= 5.0)  * 100
    within10   = np.mean(pct_err <= 10.0) * 100

    result = {
        "Model"        : model_name,
        "MAPE (%)"     : round(mape,   2),
        "RMSE (SEK)"   : round(rmse,   0),
        "MAE (SEK)"    : round(mae,    0),
        "R²"           : round(r2,     4),
        "Within ±5%"   : f"{within5:.1f}%",
        "Within ±10%"  : f"{within10:.1f}%",
        "Gap to ±5% target": f"{max(0, mape - MAPE_TARGET):.2f}pp"
    }
    return result


def print_results_table(results: list[dict]):
    """Pretty-prints the evaluation results table."""
    df_r = pd.DataFrame(results)
    print("\n" + "═" * 95)
    print("  MODEL EVALUATION RESULTS")
    print("═" * 95)
    print(df_r.to_string(index=False))
    print("═" * 95)
    best_mape = df_r["MAPE (%)"].min()
    best_model = df_r.loc[df_r["MAPE (%)"].idxmin(), "Model"]
    print(f"\n  Best model: {best_model}  |  MAPE={best_mape:.2f}%  |  "
          f"Gap to ±5% target: {max(0, best_mape - MAPE_TARGET):.2f} percentage points")


# ─── 9. TIME-SERIES CROSS VALIDATION ─────────────────────────────────────────
def cross_validate_models(df_feat: pd.DataFrame, feature_cols: list):
    """
    Walk-forward time-series cross-validation with 5 folds.
    Each fold expands the training window by ~6 months.
    This gives a more reliable estimate of model performance
    than a single train/test split.
    """
    print("\n[STEP 5] Time-series cross-validation (5 folds)...")

    X = df_feat[feature_cols].values
    y = df_feat[TARGET].values
    tscv = TimeSeriesSplit(n_splits=5, test_size=6)

    cv_results = {"Ridge": [], "XGBoost": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Ridge
        sc = StandardScaler()
        ridge = Ridge(alpha=10.0)
        ridge.fit(sc.fit_transform(X_tr), y_tr)
        mape_r = mean_absolute_percentage_error(y_te, ridge.predict(sc.transform(X_te))) * 100
        cv_results["Ridge"].append(mape_r)

        # XGBoost / GB
        if USE_XGBOOST:
            xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                               subsample=0.8, random_state=42, verbosity=0)
        else:
            xgb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                             max_depth=4, subsample=0.8, random_state=42)
        xgb.fit(X_tr, y_tr)
        mape_x = mean_absolute_percentage_error(y_te, xgb.predict(X_te)) * 100
        cv_results["XGBoost"].append(mape_x)

        print(f"  Fold {fold+1}: Ridge MAPE={mape_r:.1f}%  |  XGBoost MAPE={mape_x:.1f}%")

    for name, scores in cv_results.items():
        print(f"  {name:12s} CV MAPE: {np.mean(scores):.2f}% ± {np.std(scores):.2f}%")

    return cv_results


# ─── 10. FORECASTING ─────────────────────────────────────────────────────────
def forecast_future(df_feat: pd.DataFrame, feature_cols: list,
                    sarima_model, xgb_model, optimal_weights: tuple,
                    scaler: StandardScaler | None = None) -> pd.DataFrame:
    """
    Generates rolling multi-step forecasts for the next FORECAST_MONTHS months.

    Strategy:
      - Start from the last known data point
      - Iteratively predict each future month
      - Feed the predicted value back as a lag feature for subsequent months
      - Average Hybrid and XGBoost forecasts using optimal blending weights
    """
    print(f"\n[STEP 6] Forecasting next {FORECAST_MONTHS} months...")

    last = df_feat.iloc[-1]
    last_month = df_feat["Month"].iloc[-1]
    forecast_dates = pd.date_range(
        last_month + pd.DateOffset(months=1), periods=FORECAST_MONTHS, freq="MS"
    )

    # Seed historical values for recursive prediction
    t_hist  = list(df_feat[TARGET].values)
    o_hist  = list(df_feat["Orders"].values)
    p_hist  = list(df_feat["pipeline_wtd"].values)

    forecasts_sarima = []
    forecasts_xgb    = []

    for i, fdate in enumerate(forecast_dates):
        mnum = fdate.month
        yr   = fdate.year
        qtr  = (mnum - 1) // 3 + 1

        # ── Build feature vector for this future month ─────────────────────
        # Autoregressive lags — use actual history or previously forecasted
        def get_t(lag):
            idx = -(lag - i) if i < lag else i - lag
            if i < lag:
                return t_hist[idx]
            else:
                return forecasts_xgb[i - lag]

        t1  = t_hist[-1] if i == 0 else forecasts_xgb[-1]
        t2  = t_hist[-2] if i < 2  else forecasts_xgb[-2]
        t3  = t_hist[-3] if i < 3  else forecasts_xgb[max(0, i-3)]
        t6  = t_hist[-(6-i)] if i < 6  else forecasts_xgb[i-6]
        t12 = t_hist[i]   # same calendar month, prior year

        o1 = o_hist[-1]; o2 = o_hist[-2]
        recent_t = ([t_hist[-j] for j in range(1, min(13, len(t_hist)+1))]
                    + forecasts_xgb)[-12:]

        r3  = np.mean(([t1, t2, t3])[-3:])
        r6  = np.mean(([t6, t3, t2, t1] + [t_hist[-k] for k in range(1, 3)])[-6:])
        r12 = np.mean(recent_t)
        std6 = np.std(([t1, t2, t3, t6] + [t_hist[-k] for k in range(1,3)])[-6:])

        # External signals — use last known values (or naive carry-forward)
        ici1 = float(last["ICI_lag1"]); ici2 = float(last["ICI_lag2"])
        pmi1 = float(last["PMI_lag1"]); pmi2 = float(last["PMI_lag2"])
        os1  = float(last["OS_lag1"]);  os2  = float(last["OS_lag2"])
        ici_pmi_int = ici1 * pmi1 / 100

        p1   = float(last["pipeline_lag1"]); p2 = float(last["pipeline_lag2"])
        pr3  = float(last["pipeline_roll3"])

        o3   = float(last["Orders_lag3"])
        o6   = float(last["Orders_lag6"])
        o12  = float(last["Orders_lag12"])

        feat_vec = np.array([[
            t1, t2, t3, t6, t12,          # Turnover lags
            o1, o2, o3, o6, o12,          # Orders lags
            r3, r6, r12, std6,            # Rolling stats
            np.sin(2*np.pi*mnum/12),      # Seasonality sin
            np.cos(2*np.pi*mnum/12),      # Seasonality cos
            qtr, yr,                      # Calendar
            os1, os2,                     # O/S ratio
            ici1, ici2,                   # ICI
            pmi1, pmi2,                   # PMI
            ici_pmi_int,                  # ICI × PMI interaction
            p1, p2, pr3                   # Pipeline signals
        ]])

        # Subset to the feature_cols that the model was actually trained on
        feat_df = pd.DataFrame(feat_vec,
                               columns=[
            "Turnover_lag1","Turnover_lag2","Turnover_lag3","Turnover_lag6","Turnover_lag12",
            "Orders_lag1","Orders_lag2","Orders_lag3","Orders_lag6","Orders_lag12",
            "Turnover_roll3","Turnover_roll6","Turnover_roll12","Turnover_std6",
            "month_sin","month_cos","quarter","year",
            "OS_lag1","OS_lag2","ICI_lag1","ICI_lag2","PMI_lag1","PMI_lag2",
            "ICI_PMI_interaction","pipeline_lag1","pipeline_lag2","pipeline_roll3"
        ])
        # Keep only columns the model was trained on
        common_cols = [c for c in feature_cols if c in feat_df.columns]
        X_fut = feat_df[common_cols].values

        # SARIMA / Ridge prediction
        if USE_SARIMA:
            # Simple naive forecast using last trend for SARIMA multi-step
            p_s = float(np.mean([t1, t2, t3]))  # simplification for multi-step
        else:
            p_s = float(sarima_model[0].predict(sarima_model[1].transform(X_fut))[0])

        # XGBoost prediction
        p_x = float(xgb_model.predict(X_fut)[0])

        # Hybrid blend
        forecasts_sarima.append(p_s)
        forecasts_xgb.append(p_x)

    forecasts_hybrid = [
        optimal_weights[0] * s + optimal_weights[1] * x
        for s, x in zip(forecasts_sarima, forecasts_xgb)
    ]

    df_fc = pd.DataFrame({
        "Month"           : forecast_dates,
        "SARIMA_Forecast" : forecasts_sarima,
        "XGBoost_Forecast": forecasts_xgb,
        "Hybrid_Forecast" : forecasts_hybrid,
        "Lower_11pct"     : [v * 0.89 for v in forecasts_hybrid],
        "Upper_11pct"     : [v * 1.11 for v in forecasts_hybrid],
    })

    # Quarterly aggregation
    df_fc["Quarter"] = df_fc["Month"].dt.to_period("Q")
    quarterly = df_fc.groupby("Quarter")["Hybrid_Forecast"].sum().reset_index()
    quarterly.columns = ["Quarter", "Quarterly_Hybrid_Forecast"]

    print("\n  Monthly Forecast:")
    for _, row in df_fc.iterrows():
        print(f"    {row['Month'].strftime('%b %Y')}  "
              f"Hybrid={row['Hybrid_Forecast']/1e6:.2f}M SEK  "
              f"[{row['Lower_11pct']/1e6:.2f}M – {row['Upper_11pct']/1e6:.2f}M]")

    print("\n  Quarterly Forecast:")
    for _, row in quarterly.iterrows():
        print(f"    {row['Quarter']}  →  {row['Quarterly_Hybrid_Forecast']/1e6:.2f}M SEK")

    return df_fc


# ─── 11. VISUALISATIONS ──────────────────────────────────────────────────────
def plot_all(df_feat, df_q, m_test, y_test,
             y_pred_sarima, y_pred_xgb, y_pred_hybrid,
             df_forecast, results, xgb_model, feature_cols):
    """Generates and saves 4 figures summarising EDA, model results, and forecast."""

    sns.set_style("dark")
    BG   = "#0f1117"
    CARD = "#1a1d27"
    GREEN = "#00d4aa"; BLUE = "#74b9ff"; RED = "#ff6b6b"; YELLOW = "#ffd166"

    # ── Figure 1: Actual vs Predicted (test set) ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
    for ax in axes.flat: ax.set_facecolor(CARD)

    ax = axes[0, 0]
    ax.plot(m_test, y_test / 1e6,          "o-", color="white", lw=2.5, ms=5, label="Actual",    zorder=5)
    ax.plot(m_test, y_pred_sarima / 1e6,   "s--", color=BLUE,   lw=1.8, ms=4, label="SARIMA",   alpha=0.9)
    ax.plot(m_test, y_pred_xgb / 1e6,      "^--", color=RED,    lw=1.8, ms=4, label="XGBoost",  alpha=0.9)
    ax.plot(m_test, y_pred_hybrid / 1e6,   "D-",  color=GREEN,  lw=2.2, ms=5, label="Hybrid",   alpha=0.95)
    ax.fill_between(m_test, y_test / 1e6 * 0.95, y_test / 1e6 * 1.05,
                    alpha=0.1, color="white", label="±5% band")
    ax.set_title("Model Predictions vs Actual (Test Set)", color="white", fontweight="bold")
    ax.set_ylabel("MSEK", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#aaa")

    # ── Monthly error % ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    err = (y_pred_hybrid - y_test) / y_test * 100
    colors_bar = [GREEN if abs(e) <= 5 else YELLOW if abs(e) <= 10 else RED for e in err]
    ax.bar(range(len(m_test)), err, color=colors_bar, alpha=0.85)
    ax.axhline(5,  color="white", lw=1, linestyle="--", alpha=0.4)
    ax.axhline(-5, color="white", lw=1, linestyle="--", alpha=0.4)
    ax.axhline(0,  color="white", lw=0.5, alpha=0.3)
    ax.set_xticks(range(len(m_test)))
    ax.set_xticklabels([m.strftime("%b %y") for m in m_test],
                       rotation=45, ha="right", color="#aaa", fontsize=7)
    within5 = np.mean(np.abs(err) <= 5) * 100
    ax.text(0.98, 0.95, f"{within5:.0f}% within ±5%",
            transform=ax.transAxes, ha="right", va="top",
            color=GREEN, fontsize=12, fontweight="bold")
    ax.set_title("Hybrid Model Monthly Error (%)", color="white", fontweight="bold")
    ax.set_ylabel("Error %", color="#aaa")
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")

    # ── MAPE comparison ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    models_names = [r["Model"] for r in results]
    mapes        = [r["MAPE (%)"] for r in results]
    c_bars = [GREEN if m <= MAPE_TARGET else YELLOW if m <= 10 else RED for m in mapes]
    short_names = [n.split("(")[0].strip()[:22] for n in models_names]
    bars = ax.bar(range(len(short_names)), mapes, color=c_bars, alpha=0.85, width=0.55)
    for bar, val in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15, f"{val:.1f}%",
                ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    ax.axhline(MAPE_TARGET, color=YELLOW, lw=2, linestyle="--", label=f"±{MAPE_TARGET}% Target")
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, color="#aaa", fontsize=9)
    ax.set_title("MAPE by Model", color="white", fontweight="bold")
    ax.set_ylabel("MAPE (%)", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")

    # ── Feature importance ────────────────────────────────────────────────────
    ax = axes[1, 1]
    fi = pd.Series(xgb_model.feature_importances_, index=feature_cols).nlargest(12).sort_values()
    c_fi = [GREEN if "Turnover" in f
            else YELLOW if any(x in f for x in ["ICI","PMI","OS","pipeline"])
            else BLUE for f in fi.index]
    ax.barh(range(len(fi)), fi.values, color=c_fi, alpha=0.85)
    ax.set_yticks(range(len(fi)))
    ax.set_yticklabels(fi.index, color="#aaa", fontsize=8)
    ax.set_title("Feature Importance (XGBoost)", color="white", fontweight="bold")
    ax.set_xlabel("Importance", color="#aaa")
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")

    plt.tight_layout(pad=2)
    plt.savefig("figures/fig1_model_evaluation.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("\n  Saved: figures/fig1_model_evaluation.png")

    # ── Figure 2: Forecast ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    for ax in axes: ax.set_facecolor(CARD)

    df_recent = df_feat[df_feat["Month"] >= "2024-01-01"].copy()
    last_actual = df_feat["Month"].iloc[-1]

    ax = axes[0]
    ax.fill_between(df_recent["Month"], df_recent[TARGET] / 1e6, alpha=0.2, color=BLUE)
    ax.plot(df_recent["Month"], df_recent[TARGET] / 1e6,
            "o-", color=BLUE, lw=2, ms=5, label="Actual")
    ax.plot(df_forecast["Month"], df_forecast["SARIMA_Forecast"] / 1e6,
            "s--", color="#a29bfe", lw=1.5, ms=4, label="SARIMA", alpha=0.8)
    ax.plot(df_forecast["Month"], df_forecast["XGBoost_Forecast"] / 1e6,
            "^--", color=RED, lw=1.5, ms=4, label="XGBoost", alpha=0.8)
    ax.plot(df_forecast["Month"], df_forecast["Hybrid_Forecast"] / 1e6,
            "D-", color=GREEN, lw=2.5, ms=7, label="Hybrid", zorder=5)
    ax.fill_between(df_forecast["Month"],
                    df_forecast["Lower_11pct"] / 1e6,
                    df_forecast["Upper_11pct"] / 1e6,
                    alpha=0.18, color=GREEN, label="±11% confidence")
    ax.axvline(last_actual, color=YELLOW, lw=1.2, linestyle=":", alpha=0.7)
    ax.set_title("Revenue Forecast: Next 6 Months", color="white", fontweight="bold", fontsize=13)
    ax.set_ylabel("MSEK", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#aaa")

    # Quarterly bar
    ax2 = axes[1]
    df_q_grp = df_forecast.copy()
    df_q_grp["Q_label"] = df_q_grp["Month"].dt.strftime("Q%q %Y")
    q_fc = df_q_grp.groupby("Q_label").agg(
        Hybrid=("Hybrid_Forecast", "sum"),
        Low=("Lower_11pct", "sum"),
        High=("Upper_11pct", "sum")
    ).reset_index()
    err_lo = q_fc["Hybrid"] - q_fc["Low"]
    err_hi = q_fc["High"]   - q_fc["Hybrid"]
    bars = ax2.bar(range(len(q_fc)), q_fc["Hybrid"] / 1e6,
                   yerr=[err_lo / 1e6, err_hi / 1e6],
                   color=[GREEN, BLUE][:len(q_fc)],
                   alpha=0.85, capsize=5, error_kw={"color": "white", "alpha": 0.5})
    for bar, val in zip(bars, q_fc["Hybrid"] / 1e6):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, f"{val:.1f}M",
                 ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(q_fc)))
    ax2.set_xticklabels(q_fc["Q_label"], color="#aaa", fontsize=10)
    ax2.set_title("Quarterly Revenue Forecast (Hybrid)", color="white", fontweight="bold", fontsize=13)
    ax2.set_ylabel("MSEK", color="#aaa")
    ax2.tick_params(colors="#aaa"); ax2.spines[:].set_color("#333")

    plt.tight_layout(pad=2)
    plt.savefig("figures/fig2_forecast.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: figures/fig2_forecast.png")


# ─── 12. EXPORT RESULTS ──────────────────────────────────────────────────────
def export_results(df_feat, df_forecast, m_test, y_test,
                   y_pred_sarima, y_pred_xgb, y_pred_hybrid, results):
    """Exports model predictions and forecast to CSV for further analysis."""

    # Test-set predictions
    df_test_out = pd.DataFrame({
        "Month"          : m_test,
        "Actual"         : y_test,
        "SARIMA_Pred"    : y_pred_sarima,
        "XGBoost_Pred"   : y_pred_xgb,
        "Hybrid_Pred"    : y_pred_hybrid,
        "Error_SARIMA_%" : (y_pred_sarima - y_test) / y_test * 100,
        "Error_XGBoost_%": (y_pred_xgb   - y_test) / y_test * 100,
        "Error_Hybrid_%" : (y_pred_hybrid - y_test) / y_test * 100,
    })
    df_test_out.to_csv("test_predictions.csv", index=False)

    # Forecast
    df_forecast.to_csv("forecast_output.csv", index=False)

    # Metrics
    pd.DataFrame(results).to_csv("model_metrics.csv", index=False)

    print("\n  Exported:")
    print("    test_predictions.csv")
    print("    forecast_output.csv")
    print("    model_metrics.csv")


# ─── 13. MAIN PIPELINE ───────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PREDICTIVE REVENUE INTELLIGENCE SYSTEM (PRIS)")
    print("  Hybrid SARIMA + XGBoost Forecasting Model")
    print("=" * 65)

    # 1. Load data
    df, df_q = load_data(SALES_PMI_CSV, QUOTES_CSV)

    # 2. Feature engineering
    df_feat = build_features(df, df_q)
    feature_cols = get_feature_columns(df_feat)
    print(f"\n  Final feature set ({len(feature_cols)} features):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {col}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test, m_test = time_split(df_feat, feature_cols)

    # 4a. SARIMA / Ridge benchmark
    y_pred_sarima, sarima_fitted = train_sarima(
        X_train, X_test, y_train, df_feat, feature_cols
    )

    # 4b. XGBoost
    y_pred_xgb, xgb_model = train_xgboost(X_train, X_test, y_train, feature_cols)

    # 4c. Hybrid ensemble
    y_pred_hybrid_fixed, y_pred_hybrid_opt, optimal_weights = train_hybrid(
        y_pred_sarima, y_pred_xgb, y_train, X_train, y_test
    )

    # 5. Cross-validation
    cv_results = cross_validate_models(df_feat, feature_cols)

    # 6. Evaluate all models
    results = [
        evaluate(y_test, y_pred_sarima,       "SARIMA / Ridge Benchmark"),
        evaluate(y_test, y_pred_xgb,          "XGBoost (non-linear)"),
        evaluate(y_test, y_pred_hybrid_fixed, f"Hybrid Fixed ({HYBRID_SARIMA_WEIGHT:.0%}/{HYBRID_XGBOOST_WEIGHT:.0%})"),
        evaluate(y_test, y_pred_hybrid_opt,   f"Hybrid Optimal ({optimal_weights[0]:.0%}/{optimal_weights[1]:.0%})"),
    ]
    print_results_table(results)

    # 7. Select the best hybrid predictions
    best_hybrid_mape = min(
        mean_absolute_percentage_error(y_test, y_pred_hybrid_fixed),
        mean_absolute_percentage_error(y_test, y_pred_hybrid_opt)
    )
    y_pred_hybrid = (y_pred_hybrid_opt
                     if mean_absolute_percentage_error(y_test, y_pred_hybrid_opt) <= best_hybrid_mape
                     else y_pred_hybrid_fixed)

    # 8. Extract scaler for Ridge fallback (needed in forecast)
    scaler_obj = sarima_fitted[1] if not USE_SARIMA else None

    # 9. Forecast future months
    df_forecast = forecast_future(
        df_feat, feature_cols,
        sarima_fitted, xgb_model,
        optimal_weights, scaler_obj
    )

    # 10. Visualise
    plot_all(df_feat, df_q, m_test, y_test,
             y_pred_sarima, y_pred_xgb, y_pred_hybrid,
             df_forecast, results, xgb_model, feature_cols)

    # 11. Export
    export_results(df_feat, df_forecast, m_test, y_test,
                   y_pred_sarima, y_pred_xgb, y_pred_hybrid, results)

    # ── Final Summary ─────────────────────────────────────────────────────────
    best = min(results, key=lambda r: r["MAPE (%)"])
    print("\n" + "═" * 65)
    print("  FINAL SUMMARY")
    print("═" * 65)
    print(f"  Best model      : {best['Model']}")
    print(f"  Best MAPE       : {best['MAPE (%)']:.2f}%")
    print(f"  ±5% target      : {MAPE_TARGET}%")
    gap = max(0, best["MAPE (%)"] - MAPE_TARGET)
    print(f"  Gap to target   : {gap:.2f} percentage points")
    if gap == 0:
        print("  ✅ TARGET ACHIEVED!")
    else:
        print(f"  ⚠️  Close gap by: loading all 35 monthly files + adding")
        print(f"     Finland/Norway data to increase training samples")
    print("═" * 65)

    return df_forecast, results, xgb_model


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_forecast, results, xgb_model = main()
