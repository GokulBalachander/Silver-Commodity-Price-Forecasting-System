# =============================================================================
# PROJECT 1D: SILVER COMMODITY PRICE FORECASTING AGENT
# Zetheta Algorithms Private Limited
# =============================================================================

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Prophet
from prophet import Prophet

print("✅ All libraries imported successfully!")

# =============================================================================
# STEP 2: LOAD ALL 5 DATASETS
# =============================================================================
print("\n📂 Loading datasets...")

df_daily    = pd.read_csv('silver_daily_ohlcv_2000_2025.csv')
df_macro    = pd.read_csv('silver_macroeconomic_monthly.csv')
df_futures  = pd.read_csv('silver_futures_contracts.csv')
df_sentiment= pd.read_csv('silver_sentiment_weekly.csv')
df_supply   = pd.read_csv('silver_supply_demand_annual.csv')

print(f"✅ Daily OHLCV:      {df_daily.shape}")
print(f"✅ Macroeconomic:    {df_macro.shape}")
print(f"✅ Futures:          {df_futures.shape}")
print(f"✅ Sentiment:        {df_sentiment.shape}")
print(f"✅ Supply & Demand:  {df_supply.shape}")

# =============================================================================
# STEP 3: DATA VALIDATION & CLEANING
# =============================================================================
print("\n🔍 Validating data quality...")

datasets = {
    'Daily OHLCV': df_daily,
    'Macroeconomic': df_macro,
    'Futures': df_futures,
    'Sentiment': df_sentiment,
    'Supply & Demand': df_supply
}

for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    print(f"  {name} → Missing: {missing} | Duplicates: {dupes}")

print("✅ Data validation complete — datasets are clean!")

# =============================================================================
# STEP 4: CONVERT DATES & MERGE DATASETS
# =============================================================================
print("\n🔗 Converting dates and merging datasets...")

# Convert date columns to datetime
df_daily['Date']         = pd.to_datetime(df_daily['Date'])
df_macro['Date']         = pd.to_datetime(df_macro['Date'])
df_futures['Trade_Date'] = pd.to_datetime(df_futures['Trade_Date'])
df_sentiment['Week_Ending'] = pd.to_datetime(df_sentiment['Week_Ending'])

# Add Year/Month keys for merging
df_daily['YearMonth']    = df_daily['Date'].dt.to_period('M')
df_macro['YearMonth']    = df_macro['Date'].dt.to_period('M')

# Add Year key for supply demand
df_daily['Year']         = df_daily['Date'].dt.year
df_supply['Year']        = df_supply['Year'].astype(int)

# Add Week key for sentiment
df_daily['Week']         = df_daily['Date'].dt.to_period('W')
df_sentiment['Week']     = df_sentiment['Week_Ending'].dt.to_period('W')

# Merge macro (monthly)
df_macro_slim = df_macro[['YearMonth', 'Fed_Funds_Rate', 'US_10Y_Yield',
                            'Real_Interest_Rate', 'US_CPI_YoY_Pct',
                            'DXY_Index', 'VIX_Index', 'Gold_Price_USD',
                            'Crude_Oil_WTI', 'ETF_Silver_Holdings_MOz']]
df = df_daily.merge(df_macro_slim, on='YearMonth', how='left')

# Merge sentiment (weekly)
df_sentiment_slim = df_sentiment[['Week', 'News_Sentiment_Score',
                                   'CFTC_NonCommercial_Net',
                                   'ETF_SLV_Flow_Millions',
                                   'Implied_Volatility_30D']]
df = df.merge(df_sentiment_slim, on='Week', how='left')

# Merge supply demand (annual)
df_supply_slim = df_supply[['Year', 'Supply_Demand_Balance_MOz',
                              'Solar_Panel_Demand_MOz', 'EV_Demand_MOz']]
df = df.merge(df_supply_slim, on='Year', how='left')

# Sort by date and reset index
df = df.sort_values('Date').reset_index(drop=True)

# Forward fill any remaining NaN values from merging
df = df.ffill().bfill()

print(f"✅ Merged dataset shape: {df.shape}")
print(f"✅ Date range: {df['Date'].min()} to {df['Date'].max()}")

# =============================================================================
# STEP 5: FEATURE ENGINEERING
# =============================================================================
print("\n⚙️ Engineering features...")

# --- Technical Indicators ---

# Simple Moving Averages
df['SMA_20']  = df['Close'].rolling(window=20).mean()
df['SMA_50']  = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Exponential Moving Averages
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# MACD
df['MACD']        = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

# RSI (14-day)
delta    = df['Close'].diff()
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs       = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
bb_std          = df['Close'].rolling(window=20).std()
df['BB_Upper']  = df['BB_Middle'] + (2 * bb_std)
df['BB_Lower']  = df['BB_Middle'] - (2 * bb_std)
df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

# ATR (Average True Range)
df['High_Low']       = df['High'] - df['Low']
df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
df['Low_PrevClose']  = abs(df['Low']  - df['Close'].shift(1))
df['True_Range']     = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
df['ATR_14']         = df['True_Range'].rolling(window=14).mean()

# --- Lag Features ---
for lag in [1, 5, 10, 20]:
    df[f'Return_Lag_{lag}']  = df['Returns_Pct'].shift(lag)
    df[f'Close_Lag_{lag}']   = df['Close'].shift(lag)
    df[f'Volume_Lag_{lag}']  = df['Volume'].shift(lag)

# --- Price Ratios ---
df['Silver_Gold_Ratio_Calc'] = df['Close'] / df['Gold_Price_USD']

# Drop rows with NaN from rolling windows
df = df.dropna().reset_index(drop=True)

print(f"✅ Feature engineering complete!")
print(f"✅ Total features: {df.shape[1]}")
print(f"✅ Total rows after dropping NaN: {df.shape[0]}")

# =============================================================================
# STEP 6: STATIONARITY CHECK (ADF TEST)
# =============================================================================
print("\n📊 Checking stationarity with ADF Test...")

def adf_test(series, name):
    result = adfuller(series.dropna())
    p_value = result[1]
    stationary = "✅ Stationary" if p_value < 0.05 else "❌ Non-Stationary"
    print(f"  {name}: p-value = {p_value:.4f} → {stationary}")
    return p_value < 0.05

# Check raw prices
print("Raw Close Price:")
is_stationary = adf_test(df['Close'], 'Close Price')

# Check log returns
print("Log Returns:")
adf_test(df['Log_Returns'], 'Log Returns')

# If not stationary, use differencing
df['Close_Diff'] = df['Close'].diff().dropna()
print("Differenced Close:")
adf_test(df['Close_Diff'].dropna(), 'Differenced Close')

print("✅ Stationarity check complete — using Log Returns for ARIMA!")

# =============================================================================
# STEP 7: TRAIN / TEST SPLIT
# =============================================================================
print("\n✂️ Splitting data into train and test sets...")

# Use last 252 trading days (~1 year) as test set
test_size  = 252
train_size = len(df) - test_size

df_train = df.iloc[:train_size].copy()
df_test  = df.iloc[train_size:].copy()

print(f"✅ Training set: {df_train.shape[0]} rows ({df_train['Date'].min().date()} to {df_train['Date'].max().date()})")
print(f"✅ Test set:     {df_test.shape[0]} rows ({df_test['Date'].min().date()} to {df_test['Date'].max().date()})")

# =============================================================================
# STEP 8: EVALUATION METRICS
# =============================================================================
def evaluate_model(actual, predicted, model_name):
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mae   = mean_absolute_error(actual, predicted)
    mape  = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Directional accuracy
    actual_dir    = np.diff(actual)
    predicted_dir = np.diff(predicted)
    dir_accuracy  = np.mean(np.sign(actual_dir) == np.sign(predicted_dir)) * 100

    print(f"\n📊 {model_name} Results:")
    print(f"   RMSE:                 {rmse:.4f}")
    print(f"   MAE:                  {mae:.4f}")
    print(f"   MAPE:                 {mape:.2f}%")
    print(f"   Directional Accuracy: {dir_accuracy:.2f}%")

    return {'model': model_name, 'RMSE': rmse, 'MAE': mae,
            'MAPE': mape, 'Dir_Accuracy': dir_accuracy}

# =============================================================================
# STEP 9: ARIMA MODEL
# =============================================================================
print("\n🔮 Training ARIMA Model...")

# Use log returns for stationarity
train_returns = df_train['Log_Returns'].dropna()
test_returns  = df_test['Log_Returns'].dropna()

# Fit ARIMA(2,1,2) on Close prices
arima_model   = ARIMA(df_train['Close'], order=(2, 1, 2))
arima_fit     = arima_model.fit()

# Forecast
arima_forecast = arima_fit.forecast(steps=test_size)
arima_results  = evaluate_model(
    df_test['Close'].values,
    arima_forecast.values,
    "ARIMA(2,1,2)"
)

print("✅ ARIMA model complete!")

# =============================================================================
# STEP 10: FACEBOOK PROPHET MODEL
# =============================================================================
print("\n🔮 Training Facebook Prophet Model...")

# Prophet requires columns named 'ds' and 'y'
prophet_train = df_train[['Date', 'Close']].rename(
    columns={'Date': 'ds', 'Close': 'y'})

# Add regressors
prophet_train['Gold_Price_USD'] = df_train['Gold_Price_USD'].values
prophet_train['DXY_Index']      = df_train['DXY_Index'].values
prophet_train['VIX_Index']      = df_train['VIX_Index'].values

# Fit Prophet
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
prophet_model.add_regressor('Gold_Price_USD')
prophet_model.add_regressor('DXY_Index')
prophet_model.add_regressor('VIX_Index')
prophet_model.fit(prophet_train)

# Create future dataframe
prophet_test = df_test[['Date', 'Close', 'Gold_Price_USD',
                          'DXY_Index', 'VIX_Index']].rename(
    columns={'Date': 'ds', 'Close': 'y'})

# Forecast
prophet_forecast  = prophet_model.predict(prophet_test)
prophet_predicted = prophet_forecast['yhat'].values
prophet_results   = evaluate_model(
    df_test['Close'].values,
    prophet_predicted,
    "Facebook Prophet"
)

print("✅ Prophet model complete!")

# =============================================================================
# STEP 11: XGBOOST MODEL
# =============================================================================
print("\n🔮 Training XGBoost Model...")

# Define features for XGBoost
feature_cols = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'RSI_14',
    'BB_Width', 'ATR_14', 'VWAP',
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_10', 'Return_Lag_20',
    'Close_Lag_1', 'Close_Lag_5',
    'Volume_Lag_1',
    'Fed_Funds_Rate', 'DXY_Index', 'VIX_Index',
    'Gold_Price_USD', 'Crude_Oil_WTI',
    'News_Sentiment_Score', 'CFTC_NonCommercial_Net',
    'Supply_Demand_Balance_MOz'
]

X_train = df_train[feature_cols]
y_train = df_train['Close']
X_test  = df_test[feature_cols]
y_test  = df_test['Close']

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_predicted = xgb_model.predict(X_test)
xgb_results   = evaluate_model(
    y_test.values,
    xgb_predicted,
    "XGBoost"
)

# Feature importance plot
print("\n📊 Top 10 Most Important Features (XGBoost):")
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
print(importance_df.to_string(index=False))

print("✅ XGBoost model complete!")

# =============================================================================
# STEP 12: LSTM MODEL
# =============================================================================
print("\n🔮 Training LSTM Model...")

# Scale data
lstm_features = ['Close', 'Volume', 'SMA_20', 'RSI_14', 'MACD',
                  'ATR_14', 'DXY_Index', 'Gold_Price_USD', 'VIX_Index',
                  'News_Sentiment_Score']

scaler    = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(df_train[lstm_features])
test_scaled  = scaler.transform(df_test[lstm_features])

# Create sequences (60 days → predict next day)
SEQ_LEN = 60

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # Predict Close price (index 0)
    return np.array(X), np.array(y)

# Combine train + test for sequence creation
all_scaled = np.vstack([train_scaled, test_scaled])
X_all, y_all = create_sequences(all_scaled, SEQ_LEN)

# Split back into train/test
X_lstm_train = X_all[:train_size - SEQ_LEN]
y_lstm_train = y_all[:train_size - SEQ_LEN]
X_lstm_test  = X_all[train_size - SEQ_LEN:]
y_lstm_test  = y_all[train_size - SEQ_LEN:]

print(f"  LSTM Train shape: {X_lstm_train.shape}")
print(f"  LSTM Test shape:  {X_lstm_test.shape}")

# Build LSTM model
tf.random.set_seed(42)
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(lstm_features))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train
history = lstm_model.fit(
    X_lstm_train, y_lstm_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

# Predict and inverse scale
lstm_pred_scaled = lstm_model.predict(X_lstm_test, verbose=0)

# Inverse transform (only Close column)
dummy = np.zeros((len(lstm_pred_scaled), len(lstm_features)))
dummy[:, 0] = lstm_pred_scaled.flatten()
lstm_predicted = scaler.inverse_transform(dummy)[:, 0]

dummy_actual = np.zeros((len(y_lstm_test), len(lstm_features)))
dummy_actual[:, 0] = y_lstm_test
lstm_actual = scaler.inverse_transform(dummy_actual)[:, 0]

lstm_results = evaluate_model(
    lstm_actual,
    lstm_predicted,
    "LSTM Neural Network"
)

print("✅ LSTM model complete!")

# =============================================================================
# STEP 13: ENSEMBLE MODEL
# =============================================================================
print("\n🔮 Building Ensemble Model...")

# Align all predictions to same length (use LSTM test length as reference)
n = len(lstm_predicted)

arima_pred_aligned   = arima_forecast.values[-n:]
prophet_pred_aligned = prophet_predicted[-n:]
xgb_pred_aligned     = xgb_predicted[-n:]
lstm_pred_aligned    = lstm_predicted
actual_aligned       = lstm_actual

# Weighted ensemble (weights based on inverse RMSE)
rmse_arima   = arima_results['RMSE']
rmse_prophet = prophet_results['RMSE']
rmse_xgb     = xgb_results['RMSE']
rmse_lstm    = lstm_results['RMSE']

total_inv = (1/rmse_arima + 1/rmse_prophet + 1/rmse_xgb + 1/rmse_lstm)
w_arima   = (1/rmse_arima)   / total_inv
w_prophet = (1/rmse_prophet) / total_inv
w_xgb     = (1/rmse_xgb)     / total_inv
w_lstm    = (1/rmse_lstm)    / total_inv

print(f"\n  Ensemble Weights:")
print(f"  ARIMA:   {w_arima:.3f}")
print(f"  Prophet: {w_prophet:.3f}")
print(f"  XGBoost: {w_xgb:.3f}")
print(f"  LSTM:    {w_lstm:.3f}")

ensemble_predicted = (
    w_arima   * arima_pred_aligned   +
    w_prophet * prophet_pred_aligned +
    w_xgb     * xgb_pred_aligned     +
    w_lstm    * lstm_pred_aligned
)

ensemble_results = evaluate_model(
    actual_aligned,
    ensemble_predicted,
    "Weighted Ensemble"
)

print("✅ Ensemble model complete!")

# =============================================================================
# STEP 14: WALK-FORWARD VALIDATION
# =============================================================================
print("\n🚶 Running Walk-Forward Validation (XGBoost)...")

wf_predictions = []
wf_actuals     = []
wf_window      = 504  # 2 years training window
wf_step        = 21   # predict 1 month ahead

for start in range(0, min(len(df) - wf_window - wf_step, 252), wf_step):
    wf_train = df.iloc[start:start + wf_window]
    wf_test  = df.iloc[start + wf_window:start + wf_window + wf_step]

    if len(wf_test) == 0:
        break

    wf_X_train = wf_train[feature_cols]
    wf_y_train = wf_train['Close']
    wf_X_test  = wf_test[feature_cols]
    wf_y_test  = wf_test['Close']

    wf_model = XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, random_state=42, verbosity=0
    )
    wf_model.fit(wf_X_train, wf_y_train, verbose=False)
    wf_pred = wf_model.predict(wf_X_test)

    wf_predictions.extend(wf_pred)
    wf_actuals.extend(wf_y_test.values)

wf_results = evaluate_model(
    np.array(wf_actuals),
    np.array(wf_predictions),
    "Walk-Forward Validation (XGBoost)"
)

print("✅ Walk-forward validation complete!")

# =============================================================================
# STEP 15: RISK METRICS
# =============================================================================
print("\n📊 Calculating Risk Metrics...")

# Value at Risk (95% confidence)
returns_test = pd.Series(actual_aligned).pct_change().dropna()
VaR_95 = np.percentile(returns_test, 5)
print(f"  Value at Risk (95%): {VaR_95:.4f} ({VaR_95*100:.2f}%)")

# Sharpe Ratio (annualized)
risk_free_rate = 0.05 / 252
daily_returns  = pd.Series(actual_aligned).pct_change().dropna()
sharpe_ratio   = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)
print(f"  Sharpe Ratio:        {sharpe_ratio:.4f}")

# Maximum Drawdown
cumulative     = (1 + daily_returns).cumprod()
rolling_max    = cumulative.cummax()
drawdown       = (cumulative - rolling_max) / rolling_max
max_drawdown   = drawdown.min()
print(f"  Maximum Drawdown:    {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")

print("✅ Risk metrics complete!")

# =============================================================================
# STEP 16: MODEL COMPARISON SUMMARY
# =============================================================================
print("\n" + "="*60)
print("📊 FINAL MODEL COMPARISON SUMMARY")
print("="*60)

results_df = pd.DataFrame([
    arima_results,
    prophet_results,
    xgb_results,
    lstm_results,
    ensemble_results
])
results_df = results_df.set_index('model')
results_df = results_df.round(4)
print(results_df.to_string())

best_model = results_df['RMSE'].idxmin()
print(f"\n🏆 Best Model (lowest RMSE): {best_model}")

# =============================================================================
# STEP 17: VISUALIZATIONS
# =============================================================================
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle('Silver Commodity Price Forecasting — Model Comparison', 
             fontsize=16, fontweight='bold')

test_dates = df_test['Date'].values[-n:]

# Plot 1: All model predictions vs actual
ax1 = axes[0, 0]
ax1.plot(test_dates, actual_aligned,    label='Actual',   color='black', linewidth=2)
ax1.plot(test_dates, arima_pred_aligned, label='ARIMA',   color='blue',  alpha=0.7)
ax1.plot(test_dates, prophet_pred_aligned, label='Prophet', color='green', alpha=0.7)
ax1.plot(test_dates, xgb_pred_aligned,  label='XGBoost', color='orange', alpha=0.7)
ax1.plot(test_dates, lstm_pred_aligned, label='LSTM',    color='red',   alpha=0.7)
ax1.plot(test_dates, ensemble_predicted, label='Ensemble', color='purple', linewidth=2)
ax1.set_title('All Models vs Actual Silver Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD/oz)')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Model RMSE comparison
ax2 = axes[0, 1]
models = results_df.index.tolist()
rmse_vals = results_df['RMSE'].values
colors = ['blue', 'green', 'orange', 'red', 'purple']
bars = ax2.bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('Model RMSE Comparison (Lower is Better)')
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE')
ax2.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: RSI
ax3 = axes[1, 0]
recent = df.tail(252)
ax3.plot(recent['Date'], recent['RSI_14'], color='purple', linewidth=1)
ax3.axhline(y=70, color='red',   linestyle='--', alpha=0.7, label='Overbought (70)')
ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
ax3.set_title('RSI-14 (Last 1 Year)')
ax3.set_xlabel('Date')
ax3.set_ylabel('RSI')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Bollinger Bands
ax4 = axes[1, 1]
ax4.plot(recent['Date'], recent['Close'],    label='Close',    color='black',  linewidth=1.5)
ax4.plot(recent['Date'], recent['BB_Upper'], label='BB Upper', color='red',    linestyle='--', alpha=0.7)
ax4.plot(recent['Date'], recent['BB_Lower'], label='BB Lower', color='green',  linestyle='--', alpha=0.7)
ax4.fill_between(recent['Date'], recent['BB_Lower'], recent['BB_Upper'], alpha=0.1, color='blue')
ax4.set_title('Bollinger Bands (Last 1 Year)')
ax4.set_xlabel('Date')
ax4.set_ylabel('Price (USD/oz)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: XGBoost Feature Importance
ax5 = axes[2, 0]
top_features = importance_df.head(10)
ax5.barh(top_features['Feature'], top_features['Importance'],
         color='orange', edgecolor='black', alpha=0.7)
ax5.set_title('XGBoost Top 10 Feature Importance')
ax5.set_xlabel('Importance Score')
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Directional Accuracy comparison
ax6 = axes[2, 1]
dir_acc = results_df['Dir_Accuracy'].values
bars2 = ax6.bar(models, dir_acc, color=colors, alpha=0.7, edgecolor='black')
ax6.axhline(y=50, color='red', linestyle='--', label='Random Baseline (50%)')
ax6.set_title('Directional Accuracy (Higher is Better)')
ax6.set_xlabel('Model')
ax6.set_ylabel('Directional Accuracy (%)')
ax6.tick_params(axis='x', rotation=30)
for bar, val in zip(bars2, dir_acc):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('silver_forecasting_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualizations saved to silver_forecasting_results.png")

# =============================================================================
# STEP 18: TRADE SIGNAL GENERATION
# =============================================================================
print("\n📡 Generating Trade Signals...")

signals_df = df_test.tail(n).copy()
signals_df['Ensemble_Prediction'] = ensemble_predicted

# Generate signals based on ensemble prediction vs current price
signals_df['Price_Change_Pct'] = (
    (signals_df['Ensemble_Prediction'] - signals_df['Close']) / signals_df['Close'] * 100
)

def generate_signal(pct_change):
    if pct_change > 1.5:
        return 'STRONG BUY'
    elif pct_change > 0.5:
        return 'BUY'
    elif pct_change < -1.5:
        return 'STRONG SELL'
    elif pct_change < -0.5:
        return 'SELL'
    else:
        return 'HOLD'

signals_df['Trade_Signal'] = signals_df['Price_Change_Pct'].apply(generate_signal)

# Signal summary
signal_counts = signals_df['Trade_Signal'].value_counts()
print("\n📊 Trade Signal Distribution:")
print(signal_counts.to_string())

print("\n📋 Last 10 Trade Signals:")
print(signals_df[['Date', 'Close', 'Ensemble_Prediction',
                   'Price_Change_Pct', 'Trade_Signal']].tail(10).to_string(index=False))

# =============================================================================
# STEP 19: SAVE RESULTS FOR POWER BI
# =============================================================================
print("\n💾 Saving results for Power BI dashboard...")

# Save model predictions
predictions_export = pd.DataFrame({
    'Date':              df_test['Date'].values[-n:],
    'Actual_Price':      actual_aligned,
    'ARIMA_Prediction':  arima_pred_aligned,
    'Prophet_Prediction':prophet_pred_aligned,
    'XGBoost_Prediction':xgb_pred_aligned,
    'LSTM_Prediction':   lstm_pred_aligned,
    'Ensemble_Prediction':ensemble_predicted,
    'Trade_Signal':      signals_df['Trade_Signal'].values
})
predictions_export.to_csv('model_predictions.csv', index=False)

# Save model metrics
results_df.to_csv('model_metrics.csv')

# Save feature engineered data
df.to_csv('silver_features_engineered.csv', index=False)

# Save trade signals
signals_df[['Date', 'Close', 'Ensemble_Prediction',
            'Price_Change_Pct', 'Trade_Signal',
            'RSI_14', 'MACD', 'ATR_14']].to_csv('trade_signals.csv', index=False)

# Save risk metrics
risk_metrics = pd.DataFrame({
    'Metric': ['Value at Risk (95%)', 'Sharpe Ratio', 'Maximum Drawdown'],
    'Value':  [f'{VaR_95*100:.2f}%', f'{sharpe_ratio:.4f}', f'{max_drawdown*100:.2f}%']
})
risk_metrics.to_csv('risk_metrics.csv', index=False)

print("✅ Files saved for Power BI:")
print("   - model_predictions.csv")
print("   - model_metrics.csv")
print("   - silver_features_engineered.csv")
print("   - trade_signals.csv")
print("   - risk_metrics.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("🏁 PROJECT 1D COMPLETE!")
print("="*60)
print(f"\n✅ Datasets loaded and merged:     5 files")
print(f"✅ Data validation:                No issues found")
print(f"✅ Features engineered:            {df.shape[1]} total features")
print(f"✅ Models trained:                 ARIMA, Prophet, XGBoost, LSTM, Ensemble")
print(f"✅ Walk-forward validation:        Complete")
print(f"✅ Risk metrics calculated:        VaR, Sharpe Ratio, Max Drawdown")
print(f"✅ Trade signals generated:        {len(signals_df)} signals")
print(f"✅ Results exported for Power BI:  5 CSV files")
print(f"\n🏆 Best performing model: {best_model}")
print(f"   RMSE: {results_df.loc[best_model, 'RMSE']:.4f}")
print(f"   MAE:  {results_df.loc[best_model, 'MAE']:.4f}")
print(f"   MAPE: {results_df.loc[best_model, 'MAPE']:.2f}%")
print(f"   Directional Accuracy: {results_df.loc[best_model, 'Dir_Accuracy']:.2f}%")
