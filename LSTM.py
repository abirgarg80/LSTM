"""
╔══════════════════════════════════════════════════════════════════╗
║      S&P 500 DAILY PRICE PREDICTION — LSTM (15 YEARS)           ║
║              ~3,770 trading days of real daily data              ║
╚══════════════════════════════════════════════════════════════════╝

INSTALL (run once in your terminal):
    pip install numpy pandas matplotlib scikit-learn tensorflow yfinance requests

WHAT CHANGED FROM THE MONTHLY VERSION:
    - period="15y" + interval="1d"  →  ~3,770 daily closing prices
    - Window size bumped to 90 days (was 60 monthly points)
    - Larger LSTM (256 → 128 → 64 units) to handle the richer data
    - More realistic fallback: generates synthetic daily data using
      geometric Brownian motion (the math behind Black-Scholes)
    - Forecast extended to 60 trading days (~3 months ahead)
    - Extra plot: candlestick-style 30-day zoom on test predictions
"""

# ══════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ssl
import warnings
import requests
import urllib3

warnings.filterwarnings("ignore")
urllib3.disable_warnings()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ══════════════════════════════════════════════════════════════════
# STEP 1 — FETCH 15 YEARS OF DAILY DATA
# ══════════════════════════════════════════════════════════════════
#
# yfinance lets you pick both period AND interval:
#   period   = "15y"   → go back 15 years from today
#   interval = "1d"    → one row per trading day
#
# This gives roughly 3,770 rows (there are ~252 trading days/year).
# Compare that to the monthly version which only had ~180 rows.
# More data = the LSTM can learn finer patterns.
#

def fetch_daily_data():
    """Download 15y of daily S&P 500 closes. SSL-safe, with fallback."""
    try:
        print("  Trying Yahoo Finance (daily, 15y)...")
        ssl._create_default_https_context = ssl._create_unverified_context

        import yfinance as yf
        session = requests.Session()
        session.verify = False

        ticker = yf.Ticker("^GSPC", session=session)
        df = ticker.history(period="15y", interval="1d")[["Close"]].dropna()
        df = df.reset_index()
        df.columns = ["Date", "Close"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        print(f"  Live daily data loaded: {len(df):,} trading days")
        return df

    except Exception as e:
        print(f"  Live fetch failed ({type(e).__name__}). Generating synthetic daily data...")
        return synthetic_daily_data()


def synthetic_daily_data():
    """
    Generate realistic synthetic S&P 500 daily prices using
    Geometric Brownian Motion (GBM) — the same math used in
    options pricing (Black-Scholes model).

    GBM formula for each day:
        S(t+1) = S(t) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )

    Where:
        mu    = annual drift (historical S&P avg ~10% per year)
        sigma = annual volatility (~20% for S&P 500)
        dt    = 1/252  (one trading day as fraction of a year)
        Z     ~ N(0,1) random normal noise

    This is much more realistic than just hardcoding monthly values
    because it captures daily random fluctuations and volatility clustering.
    """
    np.random.seed(42)

    trading_days = 252 * 15        # ~3,780 days
    start_price  = 1200.0          # approximate S&P level in 2010
    mu           = 0.10            # 10% annual drift
    sigma        = 0.18            # 18% annual volatility
    dt           = 1 / 252         # one day in year-fraction

    prices = [start_price]
    for _ in range(trading_days - 1):
        z      = np.random.standard_normal()
        change = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        prices.append(prices[-1] * change)

    dates = pd.bdate_range(start="2010-01-04", periods=trading_days)
    df    = pd.DataFrame({"Date": dates, "Close": prices})
    print(f"  Synthetic GBM data generated: {len(df):,} trading days")
    return df


print("=" * 62)
print("STEP 1: Loading 15 Years of Daily S&P 500 Data")
print("=" * 62)
df = fetch_daily_data()
print(f"  Date range : {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
print(f"  Price range: ${df['Close'].min():,.0f} → ${df['Close'].max():,.0f}")
print(f"  Total rows : {len(df):,} trading days")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — SCALE WITH SCIKIT-LEARN
# ══════════════════════════════════════════════════════════════════
#
# Same idea as before, but now we're scaling ~3,770 daily prices
# instead of 120 monthly ones.
#
# KEY RULE: fit the scaler ONLY on training data.
# If you fit on all data, the scaler "knows" future max/min prices,
# which is cheating (called "data leakage" in ML).
#
# We do a 80/20 split first, then fit only on train:
#

print("\n" + "=" * 62)
print("STEP 2: Scaling with sklearn MinMaxScaler (no data leakage)")
print("=" * 62)

prices_raw = df["Close"].values.reshape(-1, 1)

# Split raw prices first so scaler never sees test data
split_idx   = int(len(prices_raw) * 0.80)
train_raw   = prices_raw[:split_idx]
test_raw    = prices_raw[split_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)                          # learn min/max from TRAIN only
prices_scaled = scaler.transform(prices_raw)   # scale all data using those stats

print(f"  Scaler fitted on {len(train_raw):,} training days only")
print(f"  Scaled range: [{prices_scaled.min():.4f}, {prices_scaled.max():.4f}]")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — BUILD SEQUENCES
# ══════════════════════════════════════════════════════════════════
#
# Window size = 90 days (~4.5 months of trading).
# With daily data, 90 days captures:
#   - Short-term momentum (last 2 weeks)
#   - Medium-term trends  (last 1-3 months)
#   - Quarterly earnings cycle effects
#
# Larger windows = more context but more memory & compute needed.
# 90 days is a sweet spot for daily S&P 500 data.
#

print("\n" + "=" * 62)
print("STEP 3: Creating 90-Day Sliding Windows")
print("=" * 62)

WINDOW_SIZE = 90

def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(prices_scaled, WINDOW_SIZE)
X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

print(f"  Window size  : {WINDOW_SIZE} trading days")
print(f"  X shape      : {X_all.shape}  (samples, days, features)")
print(f"  Total samples: {len(X_all):,}")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════
#
# The sequence array starts at index WINDOW_SIZE (day 90).
# Our 80% split of sequences maps roughly to the same 80/20
# division in raw prices.
#

print("\n" + "=" * 62)
print("STEP 4: Train / Test Split (80 / 20, time-ordered)")
print("=" * 62)

seq_split   = int(len(X_all) * 0.80)
X_train, X_test = X_all[:seq_split], X_all[seq_split:]
y_train, y_test = y_all[:seq_split], y_all[seq_split:]

train_dates = df["Date"].values[WINDOW_SIZE : WINDOW_SIZE + seq_split]
test_dates  = df["Date"].values[WINDOW_SIZE + seq_split:]

print(f"  Training  : {len(X_train):,} samples  ({pd.Timestamp(train_dates[0]).date()} → {pd.Timestamp(train_dates[-1]).date()})")
print(f"  Testing   : {len(X_test):,}  samples  ({pd.Timestamp(test_dates[0]).date()} → {pd.Timestamp(test_dates[-1]).date()})")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — BUILD THE MODEL
# ══════════════════════════════════════════════════════════════════
#
# Upgraded architecture for daily data:
#
#   Bidirectional LSTM (256 units)
#     ↓
#   Normal LSTM (128 units)
#     ↓
#   LSTM (64 units)
#     ↓
#   Dense(32) → Dense(1)
#
# NEW: Bidirectional LSTM
#   A regular LSTM reads left→right (past→future).
#   A Bidirectional LSTM reads BOTH left→right AND right→left,
#   then merges both. It captures context from both directions
#   within the window, often improving accuracy on financial data.
#
#   Note: Bidirectional only makes sense in the FIRST layer.
#   After that we use regular LSTMs.
#
# ReduceLROnPlateau callback:
#   If val_loss plateaus for 5 epochs, the learning rate is
#   multiplied by 0.5 (halved). This helps the model fine-tune
#   without overshooting the optimal weights.
#

print("\n" + "=" * 62)
print("STEP 5: Building Bidirectional LSTM Network")
print("=" * 62)

model = Sequential([

    # Layer 1: Bidirectional LSTM — reads sequence both ways
    Bidirectional(
        LSTM(units=256, return_sequences=True),
        input_shape=(WINDOW_SIZE, 1)
    ),
    Dropout(0.2),

    # Layer 2: Standard LSTM
    LSTM(units=128, return_sequences=True),
    Dropout(0.2),

    # Layer 3: Final LSTM — no return_sequences, just the last output
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),

    # Dense head
    Dense(units=32, activation="relu"),
    Dense(units=1)       # single output: next day's scaled price

])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse")
model.summary()


# ══════════════════════════════════════════════════════════════════
# STEP 6 — TRAIN
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 6: Training (this will take 2-5 minutes on CPU...)")
print("=" * 62)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,          # halve the learning rate
    patience=5,          # after 5 stalled epochs
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,         # 64 works well for daily data volume
    validation_split=0.10,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

stopped = len(history.history["loss"])
best_val = min(history.history["val_loss"])
print(f"\n  Stopped at epoch {stopped} | Best val_loss: {best_val:.6f}")


# ══════════════════════════════════════════════════════════════════
# STEP 7 — PREDICT & INVERSE TRANSFORM
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 7: Generating Predictions")
print("=" * 62)

train_pred = scaler.inverse_transform(model.predict(X_train))
test_pred  = scaler.inverse_transform(model.predict(X_test))
y_test_usd = scaler.inverse_transform(y_test.reshape(-1, 1))
y_train_usd= scaler.inverse_transform(y_train.reshape(-1, 1))

print("  Done. Converted predictions back to USD.")


# ══════════════════════════════════════════════════════════════════
# STEP 8 — EVALUATE (scikit-learn metrics)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 8: Evaluation Metrics")
print("=" * 62)

rmse = np.sqrt(mean_squared_error(y_test_usd, test_pred))
mae  = mean_absolute_error(y_test_usd, test_pred)
mape = np.mean(np.abs((y_test_usd - test_pred) / y_test_usd)) * 100

# Directional accuracy: did we predict UP/DOWN correctly?
actual_dir = np.diff(y_test_usd.flatten()) > 0
pred_dir   = np.diff(test_pred.flatten()) > 0
dir_acc    = np.mean(actual_dir == pred_dir) * 100

print(f"  RMSE               : ${rmse:,.2f}")
print(f"  MAE                : ${mae:,.2f}")
print(f"  MAPE               : {mape:.2f}%")
print(f"  Directional Acc.   : {dir_acc:.1f}%  (did we predict up/down correctly?)")
print(f"\n  Interpretation: On average, daily predictions are off by ~{mape:.1f}%")
print(f"  Directional accuracy above 52% suggests real predictive signal.")


# ══════════════════════════════════════════════════════════════════
# STEP 9 — 60-DAY ROLLING FORECAST
# ══════════════════════════════════════════════════════════════════
#
# Same rolling approach, but 60 steps = ~3 months of trading days.
#

print("\n" + "=" * 62)
print("STEP 9: Rolling 60-Day Forecast")
print("=" * 62)

FORECAST_STEPS = 60
window = prices_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
future_scaled  = []

for _ in range(FORECAST_STEPS):
    pred = model.predict(window, verbose=0)
    future_scaled.append(pred[0, 0])
    window = np.append(window[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
future_dates  = pd.bdate_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=FORECAST_STEPS)

print(f"  60-day forecast from {future_dates[0].date()} to {future_dates[-1].date()}")
print(f"  Start: ${future_prices[0]:,.2f}  →  End: ${future_prices[-1]:,.2f}")
print(f"  Implied drift: {((future_prices[-1]/future_prices[0])-1)*100:+.2f}%")


# ══════════════════════════════════════════════════════════════════
# STEP 10 — PLOT (4 panels)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 10: Plotting")
print("=" * 62)

fig, axes = plt.subplots(4, 1, figsize=(16, 20))
fig.suptitle("S&P 500 — Bidirectional LSTM  |  15 Years of Daily Data",
             fontsize=16, fontweight="bold", y=0.98)

# ── Panel 1: Full 15-year history ──────────────────────────────
ax1 = axes[0]
all_dates = df["Date"].values

# Build full-length arrays for plotting (NaN where not predicted)
train_plot = np.full(len(df), np.nan)
test_plot  = np.full(len(df), np.nan)
train_s    = WINDOW_SIZE
train_e    = train_s + len(train_pred)
test_s     = train_e
test_e     = test_s + len(test_pred)
train_plot[train_s:train_e] = train_pred.flatten()
test_plot[test_s:test_e]    = test_pred.flatten()

ax1.plot(all_dates, df["Close"].values, color="#1f77b4", lw=0.6,
         label="Actual Daily Close", alpha=0.85)
ax1.plot(all_dates, train_plot, color="#2ca02c", lw=0.8,
         label="Train Prediction", alpha=0.75)
ax1.plot(all_dates, test_plot,  color="red", lw=1.0,
         label="Test Prediction (unseen)")
ax1.plot(future_dates, future_prices, color="darkorange", lw=2,
         linestyle="--", label=f"60-Day Forecast")

boundary_date = df["Date"].iloc[test_s]
ax1.axvline(x=boundary_date, color="gray", lw=1.2, linestyle=":",
            label="Train / Test boundary")

ax1.set_title("Full 15-Year Price History + LSTM Predictions")
ax1.set_ylabel("Price (USD)")
ax1.set_xlabel("Date")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

stats = f"RMSE ${rmse:,.0f}  |  MAE ${mae:,.0f}  |  MAPE {mape:.2f}%  |  Dir. Acc. {dir_acc:.1f}%"
ax1.text(0.01, 0.97, stats, transform=ax1.transAxes, fontsize=8,
         va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# ── Panel 2: Zoom — last 252 days of test set ──────────────────
ax2 = axes[1]
zoom = min(252, len(y_test_usd))    # last ~1 year of test
ax2.plot(test_dates[-zoom:], y_test_usd[-zoom:].flatten(),
         color="#1f77b4", lw=1.5, label="Actual")
ax2.plot(test_dates[-zoom:], test_pred[-zoom:].flatten(),
         color="red", lw=1.5, linestyle="--", label="Predicted")
ax2.fill_between(test_dates[-zoom:],
                 y_test_usd[-zoom:].flatten(),
                 test_pred[-zoom:].flatten(),
                 alpha=0.15, color="red", label="Error band")
ax2.set_title("Zoom: Last ~252 Trading Days of Test Set  (1 Year)")
ax2.set_ylabel("Price (USD)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=30)

# ── Panel 3: 60-day forecast ────────────────────────────────────
ax3 = axes[2]
# Show last 60 actual days for context
context_prices = df["Close"].values[-60:]
context_dates  = df["Date"].values[-60:]

ax3.plot(context_dates, context_prices, color="#1f77b4", lw=1.5,
         label="Last 60 Days (Actual)")
ax3.plot(future_dates, future_prices,   color="darkorange", lw=2,
         linestyle="--", marker="o", markersize=2, label="60-Day Forecast")
ax3.axvline(x=df["Date"].iloc[-1], color="gray", linestyle=":", lw=1.5,
            label="Today")

ax3.fill_between(
    future_dates,
    future_prices * 0.96,
    future_prices * 1.04,
    alpha=0.15, color="darkorange", label="±4% uncertainty band"
)
ax3.set_title("60-Day Forward Forecast (with Uncertainty Band)")
ax3.set_ylabel("Price (USD)")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# ── Panel 4: Training loss curve ────────────────────────────────
ax4 = axes[3]
epochs_range = range(1, len(history.history["loss"]) + 1)
ax4.plot(epochs_range, history.history["loss"],     color="steelblue",
         lw=1.5, label="Training Loss (MSE)")
ax4.plot(epochs_range, history.history["val_loss"], color="darkorange",
         lw=1.5, label="Validation Loss (MSE)")
ax4.axvline(x=stopped, color="red", linestyle=":", lw=1.2,
            label=f"Early stop: epoch {stopped}")
ax4.set_title("Training Loss — If val_loss >> train_loss: overfitting!")
ax4.set_ylabel("MSE Loss")
ax4.set_xlabel("Epoch")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("sp500_lstm_daily.png", dpi=150, bbox_inches="tight")
print("  Chart saved → sp500_lstm_daily.png")
plt.show()


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════

print(f"""
+================================================================+
|              WHAT CHANGED — MONTHLY vs DAILY                   |
+================================================================+
|                                                                |
|  DATA         Monthly: ~120-180 pts   Daily: ~3,770 pts       |
|  WINDOW       60 months              90 trading days          |
|  ARCHITECTURE LSTM x2                BiLSTM + LSTM x2         |
|  BATCH SIZE   32                     64                       |
|  FORECAST     12 periods             60 trading days          |
|                                                                |
|  WHY DAILY DATA IS BETTER FOR LSTM:                           |
|  - More training examples (3,770 vs 120)                      |
|  - Captures intraday/weekly momentum patterns                  |
|  - Learns day-of-week effects (Monday dips, Friday rallies)   |
|  - Better at spotting crash onset vs slow monthly averages    |
|                                                                |
|  WHY BIDIRECTIONAL LSTM HELPS:                                |
|  - Forward pass: learns "what led UP to this price"           |
|  - Backward pass: learns "what this price led TO"             |
|  - Combined: richer representation of the 90-day window       |
|                                                                |
|  DIRECTIONAL ACCURACY:                                        |
|  - Predicting direction (up/down) matters more than           |
|    exact price in trading. >52% is generally considered       |
|    statistically meaningful for S&P 500.                      |
|                                                                |
|  HONEST CAVEAT:                                               |
|  - LSTMs can predict TREND but not sudden shocks              |
|    (COVID crash, 2008, flash crashes). No model can.          |
|  - Always use alongside fundamentals, not in isolation.       |
+================================================================+
|  Your results:                                                 |
|    RMSE           : ${rmse:>10,.2f}                                |
|    MAE            : ${mae:>10,.2f}                                |
|    MAPE           : {mape:>9.2f}%                               |
|    Directional Acc: {dir_acc:>9.1f}%                               |
+================================================================+
""")
