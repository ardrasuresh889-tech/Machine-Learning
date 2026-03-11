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
    2. XGBoost  — Best accuracy: non-linear patterns, O/S ratio, ICI
    3. Hybrid   — Weighted ensemble of SARIMA + XGBoost

  Data:
    - 12-Month Rolling Sales vs Orders (2001–2026, ~300 months)
    - Quotation Backlog CSV (2021–2026, ~10,000 quotes)

  Author : Optima Team / Jönköping University
  Date   : March 2026
============================================================
"""

# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge

# ── Optional: real XGBoost ────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("[INFO] XGBoost found — using XGBRegressor")
except ImportError:
    USE_XGBOOST = False
    print("[INFO] XGBoost not available — using GradientBoostingRegressor (sklearn proxy)")

# ── Optional: SARIMA via statsmodels ─────────────────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    USE_SARIMA = True
    print("[INFO] statsmodels found — using SARIMAX")
except ImportError:
    USE_SARIMA = False
    print("[INFO] statsmodels not available — using Ridge SARIMA-proxy")

os.makedirs("figures", exist_ok=True)

# ─── STYLE ───────────────────────────────────────────────────────────────────
DARK_BLUE  = "#1B2A6B"
MID_BLUE   = "#2E4BB5"
ORANGE     = "#E8873A"
GREEN      = "#2ECC71"
RED        = "#E74C3C"
CREAM      = "#F5F0E8"
GRAY       = "#95A5A6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.labelcolor":  DARK_BLUE,
    "xtick.color":      DARK_BLUE,
    "ytick.color":      DARK_BLUE,
    "axes.titlecolor":  DARK_BLUE,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# !! UPDATE THESE PATHS TO MATCH YOUR COMPUTER !!
# ---------------------------------------------------------------
# Option A: Single master Excel (set USE_MONTHLY_FOLDER = False)
SALES_FILE        = r"12_Month_Rolling_salesVSOrders_TB__version_1_.xlsx"

# Option B: Folder of 35 monthly Excel files  ← schunk_260225
#   Set USE_MONTHLY_FOLDER = True and update MONTHLY_FOLDER
USE_MONTHLY_FOLDER = True
MONTHLY_FOLDER     = r"C:\Users\HP LAPTOP\Desktop\ML\final project\schunk_260225"
# ---------------------------------------------------------------

QUOTES_FILE      = r"red_quote_2021_2026.csv"
TRAIN_CUTOFF     = None               # None = auto (70% train / 30% test)
TEST_MONTHS      = 13                # months used for evaluation
FORECAST_MONTHS  = 6                 # months to forecast ahead
TARGET           = "Turnover"
MAPE_TARGET      = 5.0               # ±5% benchmark
SARIMA_WEIGHT    = 0.35
XGBOOST_WEIGHT   = 0.65

# ─── 1. DATA LOADING ─────────────────────────────────────────────────────────

# ─── 1. DATA LOADING ─────────────────────────────────────────────────────────
def _parse_num_swedish(series: pd.Series) -> pd.Series:
    """Convert Swedish number format '142 534,40' → float."""
    return (
        series.astype(str)
              .str.replace(r"\s", "", regex=True)   # remove spaces (thousands sep)
              .str.replace(",", ".", regex=False)      # Swedish decimal comma → dot
              .replace("", np.nan)
              .pipe(pd.to_numeric, errors="coerce")
    )


def _parse_one_monthly_file(filepath: str):
    """
    Each monthly CSV is a Customer Analysis snapshot file with columns:
      Company; Cust. no.; ... ; CY (ytd) - Total; ... ; PY - Total; ...
    One row = one customer. The file is named YYYY-MM.csv.

    We sum across all customers:
      Turnover = sum of "CY (ytd) - Total"  (current year to-date sales)
      Orders   = sum of "PY - Total"         (previous year total, used as order proxy)
      Month    = from filename
    """
    try:
        basename  = os.path.splitext(os.path.basename(filepath))[0]
        extension = os.path.splitext(filepath)[1].lower()

        # Month from filename
        try:
            file_month = pd.to_datetime(basename, format="%Y-%m")
        except Exception:
            print(f"    [WARN] Cannot read month from filename: {basename}")
            return None

        # Read file
        if extension == ".csv":
            df = None
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding_errors="replace",
                                     on_bad_lines="skip", low_memory=False)
                    if df.shape[1] >= 5:
                        break
                except Exception:
                    continue
            if df is None or df.empty:
                return None
        else:
            df = pd.read_excel(filepath)

        # Clean column names
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

        # ── Find Turnover column: "CY (ytd) - Total" ─────────────────────────
        turnover_col = None
        for col in df.columns:
            cl = col.lower()
            if ("cy" in cl or "current year" in cl) and "total" in cl:
                turnover_col = col
                break
        # Fallback: just "Total" column
        if turnover_col is None:
            for col in df.columns:
                if col.strip().lower() == "total":
                    turnover_col = col
                    break

        if turnover_col is None:
            return None

        # ── Find Orders column: "PY - Total" (previous year) ─────────────────
        orders_col = None
        for col in df.columns:
            cl = col.lower()
            if ("py" in cl or "previous year" in cl) and "total" in cl and "pppy" not in cl and "ppy" not in cl and "ppppy" not in cl:
                orders_col = col
                break

        # Parse and sum
        turnover = _parse_num_swedish(df[turnover_col]).fillna(0).sum()
        orders   = _parse_num_swedish(df[orders_col]).fillna(0).sum() if orders_col else turnover * 1.05

        if turnover <= 0:
            return None

        return pd.DataFrame([{
            "Month":          file_month,
            "Roll_Sales_12M": np.nan,
            "Roll_Orders_12M":np.nan,
            "OS_Ratio":       round(orders / turnover, 4) if turnover > 0 else np.nan,
            "ICI":            np.nan,
            "Turnover":       turnover,
            "Orders":         orders,
        }])

    except Exception as e:
        print(f"    [WARN] Could not parse {os.path.basename(filepath)}: {e}")
    return None


def load_sales_data(path: str = None) -> pd.DataFrame:
    print("\n[STEP 1] Loading sales & orders data …")

    if USE_MONTHLY_FOLDER:
        folder = MONTHLY_FOLDER
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"\n  ERROR: Folder not found:\n  {folder}"
                f"\n  Please update MONTHLY_FOLDER in the CONFIG section."
            )

        all_files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".csv", ".xlsx", ".xls", ".xlsm"))
        ])
        print(f"  Found {len(all_files)} data files in: {folder}")

        if len(all_files) == 0:
            raise FileNotFoundError(
                f"  No .csv or .xlsx files found in:\n  {folder}\n"
                f"  Make sure the schunk_260225 folder contains the monthly CSV files."
            )

        frames = []
        for fp in all_files:
            df_one = _parse_one_monthly_file(fp)
            if df_one is not None and len(df_one) > 0:
                frames.append(df_one)
                print(f"    OK   {os.path.basename(fp):25s}  "
                      f"Turnover = {df_one['Turnover'].iloc[0]:>15,.0f} kr")
            else:
                print(f"    SKIP {os.path.basename(fp):25s}  (no usable data)")

        if not frames:
            raise ValueError("Could not extract data from any file. Check file format.")

        df = pd.concat(frames, ignore_index=True)

    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"\n  ERROR: File not found:\n  {path}"
                f"\n  Please update SALES_FILE in the CONFIG section."
            )
        raw = pd.read_excel(path, header=None)
        df  = raw.iloc[5:, :7].copy()
        df.columns = ["Month","Roll_Sales_12M","Roll_Orders_12M",
                      "OS_Ratio","ICI","Turnover","Orders"]
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
        for col in ["Roll_Sales_12M","Roll_Orders_12M","OS_Ratio","ICI","Turnover","Orders"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate, sort
    df = df.dropna(subset=["Month","Turnover"]).copy()
    df = (
        df.groupby("Month", as_index=False)
          .agg({"Roll_Sales_12M":"last","Roll_Orders_12M":"last",
                "OS_Ratio":"last","ICI":"last","Turnover":"sum","Orders":"sum"})
    )
    df = df.sort_values("Month").reset_index(drop=True)
    print(f"\n  Combined dataset: {len(df)} months  |  "
          f"{df['Month'].min().date()} → {df['Month'].max().date()}")
    return df

def load_quotes_data(path: str) -> pd.DataFrame:
    print("[STEP 1b] Loading quotation backlog …")
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.replace("\ufeff","") for c in df.columns]

    # Parse dates (Swedish format dd.mm.yyyy)
    df["Quote Date"] = pd.to_datetime(df["Quote Date"], dayfirst=True, errors="coerce")

    # Parse value  — e.g. " 8 900 kr "
    val_col = "Value of Quotation"
    if val_col not in df.columns:
        # fuzzy match
        val_col = [c for c in df.columns if "Value" in c][0]
    df["QuoteValue"] = (
        df[val_col]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    df["Prob"] = pd.to_numeric(df["Probability (%)"], errors="coerce") / 100
    df["Converted"] = df["Status"].str.strip().str.startswith("4)").astype(int)
    df["Month"] = df["Quote Date"].dt.to_period("M").dt.to_timestamp()

    print(f"  Rows: {len(df)}  |  "
          f"{df['Quote Date'].min().date()} → {df['Quote Date'].max().date()}")
    print(f"  Conversion rate: {df['Converted'].mean():.1%}")
    return df


# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, df_q: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 2] Engineering features …")

    df = df.copy()

    # ── Time features ─────────────────────────────────────────────────────────
    df["Year"]          = df["Month"].dt.year
    df["Month_num"]     = df["Month"].dt.month
    df["Quarter"]       = df["Month"].dt.quarter
    df["Month_sin"]     = np.sin(2 * np.pi * df["Month_num"] / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df["Month_num"] / 12)
    df["Quarter_sin"]   = np.sin(2 * np.pi * df["Quarter"] / 4)
    df["Quarter_cos"]   = np.cos(2 * np.pi * df["Quarter"] / 4)

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12]:
        df[f"Turnover_lag{lag}"] = df["Turnover"].shift(lag)
        df[f"Orders_lag{lag}"]   = df["Orders"].shift(lag)

    # ── Rolling stats ─────────────────────────────────────────────────────────
    for w in [3, 6, 12]:
        df[f"Turn_rollmean{w}"]  = df["Turnover"].shift(1).rolling(w).mean()
        df[f"Turn_rollstd{w}"]   = df["Turnover"].shift(1).rolling(w).std()
        df[f"Ord_rollmean{w}"]   = df["Orders"].shift(1).rolling(w).mean()

    # ── Growth features ──────────────────────────────────────────────────────
    df["Turn_MoM"]  = df["Turnover"].pct_change(1)
    df["Turn_YoY"]  = df["Turnover"].pct_change(12)
    df["Ord_MoM"]   = df["Orders"].pct_change(1)

    # ── O/S ratio fill ────────────────────────────────────────────────────────
    df["OS_Ratio"] = df["OS_Ratio"].fillna(
        df["Orders"] / df["Turnover"].replace(0, np.nan)
    )
    df["OS_Ratio"] = df["OS_Ratio"].fillna(df["OS_Ratio"].median())

    # ── ICI fill (Industrins Konfidensindikator) ──────────────────────────────
    df["ICI"] = df["ICI"].ffill().fillna(0)

    # ── Quote backlog features ────────────────────────────────────────────────
    # Aggregate quotes per month: total value, count, weighted expected value
    q_monthly = (
        df_q.dropna(subset=["Month","QuoteValue"])
        .groupby("Month")
        .agg(
            Quote_Count   = ("QuoteValue", "count"),
            Quote_TotalVal= ("QuoteValue", "sum"),
            Quote_ExpVal  = ("QuoteValue", lambda x: (x * df_q.loc[x.index,"Prob"].fillna(0.5)).sum()),
            Quote_ConvRate= ("Converted",  "mean"),
        )
        .reset_index()
    )
    df = df.merge(q_monthly, on="Month", how="left")
    for col in ["Quote_Count","Quote_TotalVal","Quote_ExpVal","Quote_ConvRate"]:
        df[col] = df[col].fillna(0)

    # Quote lags
    df["Quote_ExpVal_lag1"] = df["Quote_ExpVal"].shift(1)
    df["Quote_ExpVal_lag2"] = df["Quote_ExpVal"].shift(2)
    df["Quote_Count_lag1"]  = df["Quote_Count"].shift(1)

    print(f"  Feature columns: {[c for c in df.columns if c not in ['Month','Turnover','Orders']]}")
    return df


# ─── 3. SARIMA PROXY ─────────────────────────────────────────────────────────
def fit_sarima(train_series: pd.Series, steps: int):
    """
    Fits SARIMAX(1,1,1)(1,1,1,12) if statsmodels available,
    otherwise falls back to Ridge with seasonal dummies + trend.
    Returns (fitted_model_or_preds, forecast_array, fitted_values).
    """
    if USE_SARIMA:
        print("  Fitting SARIMAX(1,1,1)(1,1,1,12) …")
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
        forecast = res.get_forecast(steps=steps)
        fc_vals  = forecast.predicted_mean.values
        fitted   = res.fittedvalues.values
        return res, fc_vals, fitted
    else:
        print("  Fitting Ridge-based SARIMA proxy …")
        n = len(train_series)
        y = train_series.values

        def _build_X(n_total, n_train):
            t      = np.arange(n_total)
            months = np.arange(n_total) % 12
            sin12  = np.sin(2*np.pi*months/12)
            cos12  = np.cos(2*np.pi*months/12)
            sin6   = np.sin(2*np.pi*months/6)
            cos6   = np.cos(2*np.pi*months/6)
            lag1   = np.zeros(n_total)
            lag12  = np.zeros(n_total)
            for i in range(1, n_total):
                lag1[i]  = y[i-1] if i-1 < n_train else lag1[i-1]
            for i in range(12, n_total):
                lag12[i] = y[i-12] if i-12 < n_train else lag12[i-12]
            return np.column_stack([t, t**2, sin12, cos12, sin6, cos6, lag1, lag12])

        X_train = _build_X(n, n)
        ridge = Ridge(alpha=1000)
        ridge.fit(X_train, y)

        fitted = ridge.predict(X_train)

        # Forecast
        n_total = n + steps
        y_ext   = np.concatenate([y, np.zeros(steps)])
        X_all   = _build_X(n_total, n)
        fc_all  = ridge.predict(X_all)
        fc_vals = fc_all[n:]
        return ridge, fc_vals, fitted


# ─── 4. XGBOOST / GBM MODEL ──────────────────────────────────────────────────
FEATURE_COLS = [
    "Month_sin","Month_cos","Quarter_sin","Quarter_cos","Year",
    "Turnover_lag1","Turnover_lag2","Turnover_lag3","Turnover_lag6","Turnover_lag12",
    "Orders_lag1","Orders_lag2","Orders_lag3",
    "Turn_rollmean3","Turn_rollmean6","Turn_rollmean12",
    "Turn_rollstd3","Turn_rollstd6",
    "Ord_rollmean3","Ord_rollmean6",
    "Turn_MoM","Turn_YoY","Ord_MoM",
    "OS_Ratio","ICI",
    "Quote_ExpVal_lag1","Quote_ExpVal_lag2","Quote_Count_lag1","Quote_ConvRate",
]


def fit_xgboost(X_train, y_train):
    if USE_XGBOOST:
        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )
    model.fit(X_train, y_train)
    return model


# ─── 5. EVALUATION METRICS ───────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name="Model") -> dict:
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = np.mean(np.abs(y_true - y_pred))
    r2    = r2_score(y_true, y_pred)
    bias  = np.mean(y_pred - y_true)
    within5 = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.05) * 100
    print(f"  {name:<12}  MAPE={mape:6.2f}%  RMSE={rmse:,.0f}  "
          f"MAE={mae:,.0f}  R²={r2:.3f}  Within±5%={within5:.1f}%")
    return dict(name=name, MAPE=mape, RMSE=rmse, MAE=mae, R2=r2,
                Bias=bias, Within5pct=within5)


# ─── 6. VISUALIZATION ────────────────────────────────────────────────────────
def fmt_millions(x, _):
    return f"{x/1e6:.1f}M"


def plot_eda(df: pd.DataFrame):
    """Fig 1 – Exploratory Data Analysis (4-panel)"""
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    fig.suptitle("Exploratory Data Analysis — Revenue & Orders (2001–2026)",
                 fontsize=16, fontweight="bold", color=DARK_BLUE, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: Monthly Turnover time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(df["Month"], df["Turnover"], alpha=0.15, color=MID_BLUE)
    ax1.plot(df["Month"], df["Turnover"], color=MID_BLUE, lw=1.5, label="Turnover")
    ax1.plot(df["Month"], df["Orders"],   color=ORANGE,  lw=1.2,
             linestyle="--", alpha=0.8,  label="Orders")
    ax1.set_title("Monthly Turnover & Orders (SEK)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("")
    ax1.set_ylabel("SEK (M)")

    # Panel B: Seasonality box plots
    ax2 = fig.add_subplot(gs[1, 0])
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_data = [df.loc[df["Month_num"]==m, "Turnover"].values
                    for m in range(1,13)]
    bp = ax2.boxplot(monthly_data, labels=month_names, patch_artist=True,
                     medianprops=dict(color=ORANGE, linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(MID_BLUE + "44")
    ax2.set_title("Monthly Seasonality Pattern")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax2.set_ylabel("Turnover (SEK M)")

    # Panel C: 12-Month rolling sales trend
    ax3 = fig.add_subplot(gs[1, 1])
    roll_df = df.dropna(subset=["Roll_Sales_12M"])
    ax3.plot(roll_df["Month"], roll_df["Roll_Sales_12M"],
             color=MID_BLUE, lw=2, label="Rolling Sales 12M")
    ax3.plot(roll_df["Month"], roll_df["Roll_Orders_12M"],
             color=ORANGE, lw=1.5, linestyle="--", label="Rolling Orders 12M")
    ax3.set_title("12-Month Rolling Sales vs Orders")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x/1e6:.0f}M"))
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_locator(mdates.YearLocator(4))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.set_ylabel("SEK (M)")

    plt.savefig("figures/01_EDA_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/01_EDA_overview.png")


def plot_quotes_analysis(df_q: pd.DataFrame):
    """Fig 2 – Quotation Backlog Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    fig.suptitle("Quotation Backlog Analysis (2021–2026)",
                 fontsize=16, fontweight="bold", color=DARK_BLUE)

    df_q2 = df_q.dropna(subset=["Month","QuoteValue"]).copy()

    # Monthly quote count
    ax = axes[0,0]
    mc = df_q2.groupby("Month")["QuoteValue"].count()
    ax.bar(mc.index, mc.values, color=MID_BLUE, alpha=0.8, width=25)
    ax.set_title("Monthly Quote Count")
    ax.set_ylabel("# Quotes")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Monthly quote total value
    ax = axes[0,1]
    mv = df_q2.groupby("Month")["QuoteValue"].sum() / 1e6
    ax.fill_between(mv.index, mv.values, alpha=0.25, color=ORANGE)
    ax.plot(mv.index, mv.values, color=ORANGE, lw=2)
    ax.set_title("Monthly Quote Total Value (M SEK)")
    ax.set_ylabel("M SEK")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Status breakdown
    ax = axes[1,0]
    status_counts = df_q["Status"].str.strip().value_counts()
    colors = [GREEN, RED, MID_BLUE, ORANGE, GRAY]
    wedges, texts, autotexts = ax.pie(
        status_counts.values[:5],
        labels=[s[:25] for s in status_counts.index[:5]],
        autopct="%1.1f%%",
        colors=colors[:len(status_counts)],
        startangle=140,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Quote Status Distribution")

    # Probability distribution
    ax = axes[1,1]
    prob_valid = df_q["Prob"].dropna() * 100
    ax.hist(prob_valid, bins=20, color=MID_BLUE, alpha=0.75, edgecolor="white")
    ax.axvline(prob_valid.mean(), color=ORANGE, lw=2, linestyle="--",
               label=f"Mean = {prob_valid.mean():.1f}%")
    ax.set_title("Quote Probability Distribution")
    ax.set_xlabel("Probability (%)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig("figures/02_quotes_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/02_quotes_analysis.png")


def plot_feature_correlation(df: pd.DataFrame):
    """Fig 3 – Feature correlation heatmap"""
    cols = ["Turnover","Orders","OS_Ratio","ICI",
            "Turnover_lag1","Turnover_lag12","Turn_YoY",
            "Quote_ExpVal_lag1","Quote_Count_lag1","Quote_ConvRate"]
    corr_cols = [c for c in cols if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(240, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5, ax=ax,
                cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14,
                 fontweight="bold", color=DARK_BLUE, pad=15)
    plt.tight_layout()
    plt.savefig("figures/03_feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/03_feature_correlation.png")


def plot_model_comparison(test_dates, y_test,
                          sarima_test, xgb_test, hybrid_test,
                          metrics_list):
    """Fig 4 – Model predictions vs actual on test set"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), facecolor="white")
    fig.suptitle("Model Comparison — Test Set Predictions vs Actual",
                 fontsize=15, fontweight="bold", color=DARK_BLUE)

    # ── Top: time-series predictions ─────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(test_dates, y_test, alpha=0.12, color=DARK_BLUE)
    ax.plot(test_dates, y_test, "o-", color=DARK_BLUE, lw=2, ms=5,
            zorder=5, label="Actual")
    ax.plot(test_dates, sarima_test,  "s--", color=ORANGE,   lw=1.8, ms=5,
            label="SARIMA")
    ax.plot(test_dates, xgb_test,     "^--", color=GREEN,    lw=1.8, ms=5,
            label="XGBoost/GBM")
    ax.plot(test_dates, hybrid_test,  "D-",  color=MID_BLUE, lw=2.5, ms=6,
            label="Hybrid", zorder=6)

    # ±5% band around actual
    ax.fill_between(test_dates, y_test*0.95, y_test*1.05,
                    alpha=0.07, color=GREEN, label="±5% Target Band")
    ax.set_title("Predicted vs Actual Turnover (Test Period)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Turnover (SEK M)")

    # ── Bottom: percentage error bars ────────────────────────────────────────
    ax2 = axes[1]
    w = 6  # bar width in days
    x = np.arange(len(test_dates))

    sarima_err  = (sarima_test  - y_test) / y_test * 100
    xgb_err     = (xgb_test     - y_test) / y_test * 100
    hybrid_err  = (hybrid_test  - y_test) / y_test * 100

    ax2.bar(x - w,     sarima_err,  width=w-1, label="SARIMA",       color=ORANGE,   alpha=0.8)
    ax2.bar(x,         xgb_err,     width=w-1, label="XGBoost/GBM",  color=GREEN,    alpha=0.8)
    ax2.bar(x + w,     hybrid_err,  width=w-1, label="Hybrid",       color=MID_BLUE, alpha=0.8)
    ax2.axhline(+5,  color=RED,  lw=1.5, linestyle="--", alpha=0.6)
    ax2.axhline(-5,  color=RED,  lw=1.5, linestyle="--", alpha=0.6, label="±5% target")
    ax2.axhline(0,   color=DARK_BLUE, lw=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime("%b %Y") for d in test_dates],
                        rotation=30, ha="right", fontsize=9)
    ax2.set_title("Forecast Error (%) by Month")
    ax2.set_ylabel("Error (%)")
    ax2.legend(fontsize=10)
    ax2.set_ylim(-25, 25)

    plt.tight_layout()
    plt.savefig("figures/04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/04_model_comparison.png")


def plot_metrics_dashboard(metrics_list):
    """Fig 5 – Metrics comparison bar chart"""
    df_m = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="white")
    fig.suptitle("Model Evaluation Metrics", fontsize=15,
                 fontweight="bold", color=DARK_BLUE)

    model_colors = [ORANGE, GREEN, MID_BLUE]
    bar_metrics = [
        ("MAPE",      "MAPE (%)",          True,  MAPE_TARGET),
        ("RMSE",      "RMSE (SEK)",        True,  None),
        ("R2",        "R² Score",          False, 0.90),
        ("Within5pct","Within ±5% (%)",    False, None),
    ]

    for ax, (metric, label, lower_better, threshold) in zip(axes, bar_metrics):
        bars = ax.bar(df_m["name"], df_m[metric],
                      color=model_colors[:len(df_m)], alpha=0.85, width=0.5,
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, df_m[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(df_m[metric])*0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=11, color=DARK_BLUE, fontweight="bold")
        if threshold:
            ax.axhline(threshold, color=RED, linestyle="--", lw=1.5,
                       label=f"Target={threshold}")
            ax.legend(fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)

        if metric == "RMSE":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))

    plt.tight_layout()
    plt.savefig("figures/05_metrics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/05_metrics_dashboard.png")


def plot_forecast(df: pd.DataFrame, fc_dates, fc_sarima, fc_xgb, fc_hybrid):
    """Fig 6 – 6-month forward forecast"""
    hist_window = df[df["Month"] >= "2023-01-01"].copy()

    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    fig.suptitle("6-Month Revenue Forecast (Hybrid PRIS Model)",
                 fontsize=15, fontweight="bold", color=DARK_BLUE)

    # Historical
    ax.fill_between(hist_window["Month"], hist_window["Turnover"],
                    alpha=0.10, color=DARK_BLUE)
    ax.plot(hist_window["Month"], hist_window["Turnover"],
            "o-", color=DARK_BLUE, lw=2, ms=5, label="Historical Actuals")

    # Forecasts
    ax.plot(fc_dates, fc_sarima, "s--", color=ORANGE, lw=2, ms=7,
            label="SARIMA Forecast")
    ax.plot(fc_dates, fc_xgb,    "^--", color=GREEN,  lw=2, ms=7,
            label="XGBoost/GBM Forecast")
    ax.plot(fc_dates, fc_hybrid, "D-",  color=MID_BLUE, lw=3, ms=8,
            label="Hybrid Forecast", zorder=5)

    # Confidence band (±10% around hybrid)
    ax.fill_between(fc_dates, fc_hybrid*0.90, fc_hybrid*1.10,
                    alpha=0.12, color=MID_BLUE, label="±10% Confidence Band")
    ax.fill_between(fc_dates, fc_hybrid*0.95, fc_hybrid*1.05,
                    alpha=0.18, color=MID_BLUE, label="±5% Target Band")

    # Vertical line: forecast start
    last_actual = hist_window["Month"].max()
    ax.axvline(last_actual, color=RED, linestyle=":", lw=1.5, alpha=0.7,
               label="Forecast Start")

    # Annotate forecast values
    for d, v in zip(fc_dates, fc_hybrid):
        ax.annotate(f"{v/1e6:.1f}M",
                    xy=(d, v), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color=DARK_BLUE, fontweight="bold")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Turnover (SEK M)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/06_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/06_forecast.png")


def plot_feature_importance(model, feature_names):
    """Fig 7 – Feature importance from XGBoost/GBM"""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    bars = ax.barh(np.array(feature_names)[idx],
                   importances[idx],
                   color=MID_BLUE, alpha=0.85, edgecolor="white")
    ax.set_title("Feature Importance (XGBoost/GBM) — Top 20",
                 fontsize=13, fontweight="bold", color=DARK_BLUE)
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()

    # Color top 5 differently
    for bar in bars[-5:]:
        bar.set_facecolor(ORANGE)

    plt.tight_layout()
    plt.savefig("figures/07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/07_feature_importance.png")


def plot_residuals(test_dates, y_test, hybrid_pred):
    """Fig 8 – Residual diagnostics"""
    residuals = y_test - hybrid_pred
    pct_errors = (hybrid_pred - y_test) / y_test * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.suptitle("Hybrid Model — Residual Diagnostics",
                 fontsize=14, fontweight="bold", color=DARK_BLUE)

    # Residuals over time
    ax = axes[0,0]
    ax.stem(test_dates, residuals, linefmt="b--",
            markerfmt="bo", basefmt="k-")
    ax.axhline(0, color=RED, lw=1.5)
    ax.set_title("Residuals Over Time")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Residual histogram
    ax = axes[0,1]
    ax.hist(residuals, bins=15, color=MID_BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(0, color=RED, lw=2, linestyle="--")
    ax.set_title("Residual Distribution")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax.set_ylabel("Count")

    # Actual vs Predicted scatter
    ax = axes[1,0]
    ax.scatter(y_test, hybrid_pred, color=MID_BLUE, alpha=0.8, s=70, edgecolors="white")
    lim = [min(y_test.min(), hybrid_pred.min())*0.95,
           max(y_test.max(), hybrid_pred.max())*1.05]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
    ax.fill_between(lim, np.array(lim)*0.95, np.array(lim)*1.05,
                    alpha=0.1, color=GREEN, label="±5% Band")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Scatter")
    ax.legend(fontsize=9)

    # Monthly % error
    ax = axes[1,1]
    bar_colors = [GREEN if abs(e) <= 5 else RED for e in pct_errors]
    ax.bar(range(len(pct_errors)), pct_errors, color=bar_colors, alpha=0.85)
    ax.axhline(+5, color=RED, lw=1.5, linestyle="--")
    ax.axhline(-5, color=RED, lw=1.5, linestyle="--", label="±5% Target")
    ax.axhline(0,  color=DARK_BLUE, lw=1)
    ax.set_xticks(range(len(test_dates)))
    ax.set_xticklabels([d.strftime("%b %y") for d in test_dates],
                       rotation=30, ha="right", fontsize=9)
    ax.set_title("Monthly % Error (green = within ±5%)")
    ax.set_ylabel("Error (%)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/08_residual_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/08_residual_diagnostics.png")


def plot_quarterly_analysis(df: pd.DataFrame):
    """Fig 9 – Quarterly performance & seasonality decomposition"""
    df2 = df.dropna(subset=["Turnover"]).copy()
    df2["Year"]    = df2["Month"].dt.year
    df2["Quarter"] = df2["Month"].dt.quarter
    df2["Month_num"] = df2["Month"].dt.month

    qtr = df2.groupby(["Year","Quarter"])["Turnover"].sum().reset_index()
    qtr["YearQ"] = qtr["Year"].astype(str) + "-Q" + qtr["Quarter"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.suptitle("Quarterly Revenue Analysis",
                 fontsize=14, fontweight="bold", color=DARK_BLUE)

    # Annual totals
    ax = axes[0]
    annual = df2.groupby("Year")["Turnover"].sum()
    colors_bar = [MID_BLUE if y < 2024 else ORANGE for y in annual.index]
    ax.bar(annual.index, annual.values/1e6, color=colors_bar, alpha=0.85, edgecolor="white")
    ax.set_title("Annual Revenue (M SEK)")
    ax.set_ylabel("M SEK")
    ax.set_xlabel("")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Quarterly seasonal pattern
    ax2 = axes[1]
    q_avg = qtr.groupby("Quarter")["Turnover"].mean() / 1e6
    bars2 = ax2.bar(["Q1","Q2","Q3","Q4"], q_avg.values,
                    color=[MID_BLUE, GREEN, ORANGE, RED], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, q_avg.values):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.1f}M",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=DARK_BLUE)
    ax2.set_title("Average Quarterly Revenue (All Years)")
    ax2.set_ylabel("M SEK")

    # YoY growth
    ax3 = axes[2]
    annual_growth = annual.pct_change()*100
    ax3_colors = [GREEN if g >= 0 else RED for g in annual_growth.fillna(0).values]
    ax3.bar(annual_growth.index, annual_growth.fillna(0).values,
            color=ax3_colors, alpha=0.85, edgecolor="white")
    ax3.axhline(0, color=DARK_BLUE, lw=1)
    ax3.set_title("Year-over-Year Revenue Growth (%)")
    ax3.set_ylabel("Growth (%)")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("figures/09_quarterly_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/09_quarterly_analysis.png")


def plot_os_ici_analysis(df: pd.DataFrame):
    """Fig 10 – O/S Ratio & ICI leading indicator analysis"""
    df2 = df.dropna(subset=["OS_Ratio","ICI","Turnover"]).copy()
    df2 = df2[df2["Month"] >= "2002-01-01"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    fig.suptitle("O/S Ratio & ICI — Leading Indicator Analysis",
                 fontsize=14, fontweight="bold", color=DARK_BLUE)

    # O/S Ratio time series
    ax = axes[0,0]
    ax.plot(df2["Month"], df2["OS_Ratio"], color=MID_BLUE, lw=1.5)
    ax.axhline(1.0, color=RED, lw=1.5, linestyle="--", label="OS Ratio = 1.0")
    ax.fill_between(df2["Month"], df2["OS_Ratio"], 1.0,
                    where=df2["OS_Ratio"] > 1, alpha=0.15, color=GREEN)
    ax.fill_between(df2["Month"], df2["OS_Ratio"], 1.0,
                    where=df2["OS_Ratio"] < 1, alpha=0.15, color=RED)
    ax.set_title("Orders/Sales (O/S) Ratio Over Time")
    ax.set_ylabel("O/S Ratio")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))

    # ICI time series
    ax = axes[0,1]
    ax.plot(df2["Month"], df2["ICI"], color=ORANGE, lw=1.5)
    ax.axhline(0, color=RED, lw=1, linestyle="--")
    ax.fill_between(df2["Month"], df2["ICI"], 0,
                    where=df2["ICI"] > 0, alpha=0.15, color=GREEN)
    ax.fill_between(df2["Month"], df2["ICI"], 0,
                    where=df2["ICI"] < 0, alpha=0.15, color=RED)
    ax.set_title("Industrial Confidence Indicator (ICI)")
    ax.set_ylabel("ICI Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))

    # OS Ratio vs Turnover scatter
    ax = axes[1,0]
    sc = ax.scatter(df2["OS_Ratio"], df2["Turnover"]/1e6,
                    c=df2["Year"], cmap="viridis", alpha=0.6, s=30)
    r, p = pearsonr(df2["OS_Ratio"], df2["Turnover"])
    ax.set_title(f"O/S Ratio vs Turnover  (r = {r:.2f})")
    ax.set_xlabel("O/S Ratio")
    ax.set_ylabel("Turnover (M SEK)")
    plt.colorbar(sc, ax=ax, label="Year")

    # ICI vs Turnover (1-month lag)
    ax = axes[1,1]
    df2["ICI_lag1"] = df2["ICI"].shift(1)
    valid = df2.dropna(subset=["ICI_lag1","Turnover"])
    r2, p2 = pearsonr(valid["ICI_lag1"], valid["Turnover"])
    ax.scatter(valid["ICI_lag1"], valid["Turnover"]/1e6,
               c=valid["Year"], cmap="plasma", alpha=0.5, s=30)
    ax.set_title(f"ICI(t-1) vs Turnover(t)  (r = {r2:.2f})")
    ax.set_xlabel("ICI (1-month lag)")
    ax.set_ylabel("Turnover (M SEK)")

    plt.tight_layout()
    plt.savefig("figures/10_os_ici_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/10_os_ici_analysis.png")


def plot_summary_dashboard(metrics_list, fc_dates, fc_hybrid, df):
    """Fig 11 – Executive summary dashboard"""
    fig = plt.figure(figsize=(20, 12), facecolor=CREAM)
    fig.suptitle("PREDICTIVE REVENUE INTELLIGENCE SYSTEM (PRIS)\nExecutive Summary Dashboard",
                 fontsize=18, fontweight="bold", color=DARK_BLUE, y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # KPI boxes (top row)
    df_m = pd.DataFrame(metrics_list)
    best_idx = df_m["MAPE"].idxmin()
    best = df_m.iloc[best_idx]

    kpi_data = [
        ("Best MAPE", f"{best['MAPE']:.2f}%", best['MAPE'] <= MAPE_TARGET),
        ("Best R²",   f"{df_m['R2'].max():.3f}", df_m['R2'].max() >= 0.85),
        ("Within ±5%",f"{best['Within5pct']:.1f}%", best['Within5pct'] >= 60),
        ("Model",     str(best['name']), True),
    ]
    kpi_labels = ["Best MAPE", "Best R²", "Within ±5%", "Best Model"]
    for i, (label, value, good) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(GREEN + "22" if good else RED + "22")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis("off")
        ax.text(0.5, 0.65, value, ha="center", va="center",
                fontsize=22 if i < 3 else 16,
                fontweight="bold", color=DARK_BLUE)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=10, color=GRAY)
        rect = plt.Rectangle((0.02,0.02), 0.96, 0.96, fill=False,
                              edgecolor=GREEN if good else RED, lw=2)
        ax.add_patch(rect)

    # Recent history + forecast
    ax2 = fig.add_subplot(gs[1, :])
    hist = df[df["Month"] >= "2022-01-01"].dropna(subset=["Turnover"])
    ax2.fill_between(hist["Month"], hist["Turnover"], alpha=0.10, color=DARK_BLUE)
    ax2.plot(hist["Month"], hist["Turnover"], "o-", color=DARK_BLUE, lw=2,
             ms=4, label="Actuals")
    ax2.plot(fc_dates, fc_hybrid, "D-", color=MID_BLUE, lw=2.5, ms=7,
             label="Hybrid Forecast")
    ax2.fill_between(fc_dates, fc_hybrid*0.95, fc_hybrid*1.05,
                     alpha=0.2, color=MID_BLUE)
    ax2.axvline(hist["Month"].max(), color=RED, linestyle=":", lw=1.5, alpha=0.6)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(fmt_millions))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_ylabel("Turnover (SEK M)")
    ax2.set_title("Recent History + 6-Month Hybrid Forecast (shaded = ±5%)", pad=8)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Metrics table
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    table_data = [[m["name"], f"{m['MAPE']:.2f}%",
                   f"{m['RMSE']:,.0f}", f"{m['MAE']:,.0f}",
                   f"{m['R2']:.3f}", f"{m['Within5pct']:.1f}%"]
                  for m in metrics_list]
    col_labels = ["Model", "MAPE", "RMSE", "MAE", "R²", "Within ±5%"]
    table = ax3.table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(DARK_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == best_idx + 1:
            cell.set_facecolor(MID_BLUE + "33")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#DDDDDD")
    ax3.set_title("Model Comparison Summary", fontsize=12,
                  fontweight="bold", color=DARK_BLUE, pad=8)

    plt.savefig("figures/11_executive_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/11_executive_dashboard.png")


# ─── 7. MAIN PIPELINE ────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  PREDICTIVE REVENUE INTELLIGENCE SYSTEM (PRIS)")
    print("="*65)

    # ── Load data ─────────────────────────────────────────────────────────────
    df_raw  = load_sales_data(SALES_FILE if not USE_MONTHLY_FOLDER else None)
    df_q    = load_quotes_data(QUOTES_FILE)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = engineer_features(df_raw, df_q)

    # ── EDA plots (generated before modelling) ────────────────────────────────
    print("\n[STEP 3] Generating EDA visualizations …")
    df["Month_num"] = df["Month"].dt.month   # ensure present before plot_eda
    plot_eda(df)
    plot_quotes_analysis(df_q)
    plot_feature_correlation(df)
    plot_quarterly_analysis(df)
    plot_os_ici_analysis(df)

    # ── Train / test split ────────────────────────────────────────────────────
    print("\n[STEP 4] Splitting data …")

    # Use all rows that have no NaN in features or target
    df_model = df.dropna(subset=[TARGET]).copy()

    # Fill NaN features with column median before splitting
    for col in FEATURE_COLS:
        if col in df_model.columns:
            df_model[col] = df_model[col].replace([np.inf, -np.inf], np.nan)
            df_model[col] = df_model[col].fillna(df_model[col].median())

    df_model = df_model.dropna(subset=FEATURE_COLS)
    df_model = df_model.sort_values("Month").reset_index(drop=True)

    # Dynamic cutoff: use 70% of data for training
    if TRAIN_CUTOFF is None:
        cutoff_idx = max(1, int(len(df_model) * 0.70))
        cutoff_date = df_model["Month"].iloc[cutoff_idx - 1]
    else:
        cutoff_date = pd.to_datetime(TRAIN_CUTOFF)

    train_df = df_model[df_model["Month"] <= cutoff_date]
    test_df  = df_model[df_model["Month"] >  cutoff_date]

    if len(train_df) < 2:
        raise ValueError(
            f"Not enough training data ({len(train_df)} rows). "
            f"Check that your monthly CSV files loaded correctly."
        )
    if len(test_df) < 1:
        # If no test data, use last 20% of train as test
        split = max(1, int(len(train_df) * 0.80))
        test_df  = train_df.iloc[split:].copy()
        train_df = train_df.iloc[:split].copy()

    print(f"  Train: {len(train_df)} months  ({train_df['Month'].min().date()} → {train_df['Month'].max().date()})")
    print(f"  Test:  {len(test_df)}  months  ({test_df['Month'].min().date()} → {test_df['Month'].max().date()})")

    y_train = train_df[TARGET].values
    X_train = train_df[FEATURE_COLS].values
    y_test  = test_df[TARGET].values
    X_test  = test_df[FEATURE_COLS].values
    test_dates = pd.to_datetime(test_df["Month"].values)

    # ── Scaling for XGBoost ───────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: SARIMA
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 5a] SARIMA …")
    train_series = pd.Series(
        y_train,
        index=pd.date_range(train_df["Month"].min(), periods=len(y_train), freq="MS")
    )
    sarima_model, _, sarima_fitted = fit_sarima(train_series, steps=len(y_test)+FORECAST_MONTHS)
    sarima_test = sarima_fitted[-len(y_test):] if len(sarima_fitted) >= len(y_test) else \
                  np.full(len(y_test), np.mean(y_train))

    # If SARIMA returned forecast use those for test
    if USE_SARIMA:
        _, sarima_fc_all, _ = fit_sarima(train_series, steps=len(y_test)+FORECAST_MONTHS)
        sarima_test         = sarima_fc_all[:len(y_test)]
        sarima_forecast_6m  = sarima_fc_all[len(y_test):]
    else:
        sarima_test         = sarima_fitted[-len(y_test):]
        # Re-fit for pure forecast
        _, sarima_fc_all, _ = fit_sarima(train_series, steps=len(y_test)+FORECAST_MONTHS)
        sarima_forecast_6m  = sarima_fc_all[len(y_test):]

    metrics_sarima = compute_metrics(y_test, sarima_test, "SARIMA")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: XGBoost / GBM
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 5b] XGBoost / GBM …")
    xgb_model   = fit_xgboost(X_tr_sc, y_train)
    xgb_test    = xgb_model.predict(X_te_sc)
    metrics_xgb = compute_metrics(y_test, xgb_test, "XGBoost/GBM")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 3: HYBRID
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 5c] Hybrid (weighted ensemble) …")
    hybrid_test    = SARIMA_WEIGHT * sarima_test + XGBOOST_WEIGHT * xgb_test
    metrics_hybrid = compute_metrics(y_test, hybrid_test, "Hybrid")

    metrics_list = [metrics_sarima, metrics_xgb, metrics_hybrid]

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST: 6 months ahead
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 6] Generating 6-month forward forecast …")
    last_date  = df.dropna(subset=["Turnover"])["Month"].max()
    fc_dates   = pd.date_range(last_date + pd.DateOffset(months=1),
                               periods=FORECAST_MONTHS, freq="MS")

    # XGBoost forecast: walk-forward
    recent = df.dropna(subset=FEATURE_COLS + [TARGET]).tail(24).copy()
    xgb_fc = []
    for i in range(FORECAST_MONTHS):
        last_row = recent.iloc[-1:].copy()
        new_date = last_date + pd.DateOffset(months=i+1)

        # Update time features
        last_row = last_row.copy()
        last_row["Month"]      = new_date
        last_row["Year"]       = new_date.year
        last_row["Month_num"]  = new_date.month
        last_row["Quarter"]    = (new_date.month - 1) // 3 + 1
        last_row["Month_sin"]  = np.sin(2*np.pi*new_date.month/12)
        last_row["Month_cos"]  = np.cos(2*np.pi*new_date.month/12)

        x_fc = scaler.transform(last_row[FEATURE_COLS].values)
        pred = xgb_model.predict(x_fc)[0]
        xgb_fc.append(pred)

        # Append to rolling window for next step's lags
        last_row[TARGET] = pred
        recent = pd.concat([recent, last_row], ignore_index=True)

    xgb_forecast_6m    = np.array(xgb_fc)
    sarima_forecast_6m = sarima_forecast_6m[:FORECAST_MONTHS]

    hybrid_forecast_6m = (SARIMA_WEIGHT * sarima_forecast_6m +
                          XGBOOST_WEIGHT * xgb_forecast_6m)

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  6-Month Hybrid Revenue Forecast                    │")
    print("  ├──────────────┬──────────────┬────────────────────── │")
    print("  │ Month        │ Hybrid (SEK) │ ±5% Range             │")
    print("  ├──────────────┼──────────────┼────────────────────── │")
    for d, v in zip(fc_dates, hybrid_forecast_6m):
        lo, hi = v*0.95, v*1.05
        print(f"  │ {d.strftime('%b %Y'):<12} │ {v:>12,.0f} │ {lo:,.0f} – {hi:,.0f} │")
    print("  └──────────────┴──────────────┴────────────────────── ┘")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 7] Generating model comparison & diagnostic plots …")
    plot_model_comparison(test_dates, y_test, sarima_test, xgb_test, hybrid_test, metrics_list)
    plot_metrics_dashboard(metrics_list)
    plot_forecast(df, fc_dates, sarima_forecast_6m, xgb_forecast_6m, hybrid_forecast_6m)
    plot_feature_importance(xgb_model, FEATURE_COLS)
    plot_residuals(test_dates, y_test, hybrid_test)
    plot_summary_dashboard(metrics_list, fc_dates, hybrid_forecast_6m, df)

    # ─────────────────────────────────────────────────────────────────────────
    # CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 8] Time-series cross-validation (XGBoost) …")
    df_cv = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    X_cv_arr = df_cv[FEATURE_COLS].replace([np.inf,-np.inf], np.nan).fillna(df_cv[FEATURE_COLS].median()).values
    X_cv  = scaler.fit_transform(X_cv_arr)
    y_cv  = df_cv[TARGET].values

    tscv    = TimeSeriesSplit(n_splits=5)
    cv_mape = []
    for fold, (tr, te) in enumerate(tscv.split(X_cv), 1):
        m = fit_xgboost(X_cv[tr], y_cv[tr])
        p = m.predict(X_cv[te])
        mape = mean_absolute_percentage_error(y_cv[te], p) * 100
        cv_mape.append(mape)
        print(f"  Fold {fold}: MAPE = {mape:.2f}%")
    print(f"  CV Mean MAPE: {np.mean(cv_mape):.2f}%  ±  {np.std(cv_mape):.2f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RESULTS SUMMARY")
    print("="*65)
    for m in metrics_list:
        status = "✅ TARGET MET" if m["MAPE"] <= MAPE_TARGET else "⚠️  Above target"
        print(f"  {m['name']:<15}  MAPE={m['MAPE']:5.2f}%  R²={m['R2']:.3f}  "
              f"Within±5%={m['Within5pct']:.1f}%  {status}")
    best = min(metrics_list, key=lambda x: x["MAPE"])
    print(f"\n  Best model: {best['name']}  (MAPE = {best['MAPE']:.2f}%)")
    print(f"\n  Figures saved to: ./figures/  ({len(os.listdir('figures'))} files)")
    print("="*65)

    return metrics_list, fc_dates, hybrid_forecast_6m


if __name__ == "__main__":
    main()
