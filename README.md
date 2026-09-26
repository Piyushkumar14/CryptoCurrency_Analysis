# Cryptocurrency Analysis Dashboard

A Streamlit dashboard for exploring cryptocurrency market data, technical indicators, forecast models, volatility, and sample news sentiment. Historical price data is retrieved from Yahoo Finance through `yfinance`.

## Features

- Market overview with price charts, performance metrics, trend information, and simple indicator signals
- Technical analysis using moving averages, RSI, MACD, Bollinger Bands, and rolling support and resistance
- Price forecasts using ARIMA, LSTM, or linear regression
- Volatility and risk metrics, including drawdown, value at risk, and return statistics
- Sentiment scoring for bundled sample crypto headlines with TextBlob and a small crypto-specific word list
- Comparison of multiple cryptocurrencies

## Requirements

- Python 3.9 or newer
- Internet access when retrieving price history from Yahoo Finance

Dependencies are listed in `requirements.txt`. `volatility_analyzer.py` imports SciPy, so install it explicitly if it is not installed in your environment:

```bash
pip install scipy
```

TensorFlow is included for the optional LSTM forecast. It can require additional setup and resources depending on your operating system and Python version.

## Setup

Create and activate a virtual environment, then install the dependencies:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install scipy
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install scipy
```

## Run the dashboard

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually <http://localhost:8501>) to open in your browser. Choose a cryptocurrency, time period, and analysis view from the sidebar. In the forecasting view, choose a model and forecast horizon, then select **Generate Forecast**.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and analysis views |
| `data_collector.py` | Downloads OHLCV price history from Yahoo Finance |
| `data_preprocessing.py` | Calculates technical indicators and basic signals |
| `ARIMA_model.py` | Fits and forecasts ARIMA time series models |
| `LSTM_model.py` | Prepares, trains, and forecasts with an LSTM network |
| `sentiment_analyzer.py` | Scores text and bundled example headlines |
| `volatility_analyzer.py` | Calculates volatility, risk, and performance statistics |
| `requirements.txt` | Python package requirements |

## Notes and limitations

- Yahoo Finance data availability and download behavior depend on the upstream service and network access.
- The sentiment view analyzes static example headlines in the code; it does not fetch live news.
- Forecasts and indicator signals are exploratory outputs. They are not financial advice and do not guarantee future prices or returns.
- The dashboard downloads data on demand. Longer history and LSTM training may take more time.
