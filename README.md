Silver Commodity Price Forecasting System

A comprehensive silver commodity price forecasting system built during a Data Analyst Internship at Zetheta Algorithms Private Limited. Integrates 135k+ records across 5 market datasets, engineers 64+ time-series features, and compares ARIMA, SARIMAX, Facebook Prophet, XGBoost, and LSTM ensemble models — achieving 0.49% MAPE and 80.88% directional accuracy. Results are visualized through an interactive Power BI dashboard for real-time portfolio monitoring and trade signal generation.


📊 Project Overview
Intern: Gokul Balachander
Company: Zetheta Algorithms Private Limited
Role: Data Analyst Intern 
Duration: March 2026 – May 2026 
Project: Data Analyst Commodity Price Forecasting Agent

🏆 Key Results
ModelRMSEMAEMAPEDirectional AccuracyARIMA(2,1,2)2.22751.89245.65%51.39%Facebook Prophet2.94302.43557.18%49.80%XGBoost ⭐0.21600.16620.49%80.88%LSTM Neural Network————Weighted Ensemble————

⭐ XGBoost achieved the best performance across all metrics.


📁 Project Structure
│
├── 📓 silver_forecasting.py          # Complete Python pipeline
├── 📄 Silver_Forecasting_Report.pdf  # Final project report
├── 📊 README.md                      # This file
├── 📋 requirements.txt               # Python dependencies
│
├── data/
│   ├── silver_daily_ohlcv_2000_2025.csv     # Main dataset (6,783 rows)
│   ├── silver_macroeconomic_monthly.csv     # Macro indicators (312 rows)
│   ├── silver_futures_contracts.csv         # Futures data (27,390 rows)
│   ├── silver_sentiment_weekly.csv          # Sentiment data (1,043 rows)
│   └── silver_supply_demand_annual.csv      # Supply & demand (26 rows)
│
└── outputs/
    ├── model_predictions.csv         # All model predictions vs actual
    ├── model_metrics.csv             # RMSE, MAE, MAPE, Directional Accuracy
    ├── trade_signals.csv             # 252 generated trade signals
    ├── risk_metrics.csv              # VaR, Sharpe Ratio, Max Drawdown
    └── silver_features_engineered.csv # Full dataset with 64 features

🗂️ Datasets
DatasetRowsFrequencyKey FeaturesSilver Daily OHLCV6,783DailyOpen, High, Low, Close, Volume, VWAP, ReturnsSilver Macroeconomic312MonthlyFed Rate, CPI, DXY, Gold, Oil, VIXSilver Futures27,390DailyFutures Price, Basis, Open InterestSilver Sentiment1,043WeeklyNews Sentiment, CFTC Positioning, ETF FlowsSilver Supply & Demand26AnnualMine Production, Industrial Demand, EV Demand

⚙️ Methodology
1. Data Validation

Validated 135k+ records across 5 datasets
Zero missing values and zero duplicates confirmed
Data inconsistencies reduced by 30%

2. Feature Engineering
64+ features engineered, including:

Technical Indicators: SMA (20, 50, 200), EMA (12, 26), MACD, RSI-14, Bollinger Bands, ATR-14, VWAP
Lag Features: Return_Lag_1, Return_Lag_5, Return_Lag_10, Return_Lag_20
Macroeconomic Features: Fed Funds Rate, DXY, VIX, Gold Price, Crude Oil
Sentiment Features: News Sentiment, CFTC Net Positioning, ETF Flows

3. Stationarity Testing (ADF Test)
Seriesp-valueResultRaw Close Price0.5923❌ Non-StationaryLog Returns0.0000✅ StationaryDifferenced Close0.0000✅ Stationary
4. Train/Test Split

Training: 6,332 rows (Oct 2000 – Jan 2025)
Testing: 252 rows (Jan 2025 – Dec 2025) — ~1 trading year

5. Models

ARIMA(2,1,2) — Classical time-series model
Facebook Prophet — Trend + seasonality with external regressors
XGBoost — Gradient boosted trees on 64 engineered features
LSTM — Deep learning on 60-day sequences
Weighted Ensemble — Inverse-RMSE weighted combination

6. Walk-Forward Validation

2-year rolling training window
21-day (1 month) prediction horizon
Repeated across test period


📈 Trade Signal Generation
252 trade signals generated based on ensemble predictions:
SignalThreshold🟢 STRONG BUY> +1.5% predicted change🔵 BUY+0.5% to +1.5%⚪ HOLD-0.5% to +0.5%🟠 SELL-1.5% to -0.5%🔴 STRONG SELL< -1.5%

⚠️ Risk Metrics
MetricValueValue at Risk (95%)-2.18%Sharpe Ratio0.84Maximum Drawdown-18.43%

📊 Power BI Dashboard
Interactive dashboard built with advanced DAX measures featuring:

Portfolio Overview — Key KPIs and silver price trends
Model Performance — RMSE and accuracy comparison
Trade Signal Monitor — 252 signals with price overlay
Risk Analytics — VaR, Sharpe Ratio, Drawdown analysis


🛠️ Installation & Usage
Prerequisites
bashpip install -r requirements.txt
Run the Pipeline
bashpython silver_forecasting.py
Output Files
All output CSV files will be generated in the working directory, ready for Power BI import.

👤 Author
Gokul Balachander

🎓 B.S. Computer Science with Business Applications — UC Riverside (GPA: 3.67)
💼 Data Analyst Intern @ Zetheta Algorithms Private Limited
🔗 LinkedIn: https://www.linkedin.com/in/gokul-balachander
📧 gokulbalachander.viiia@gmail.com
