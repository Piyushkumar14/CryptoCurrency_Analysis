import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

class CryptoDataCollector:
    def __init__(self):
        self.crypto_symbols = {
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'cardano': 'ADA-USD',
            'ripple': 'XRP-USD',
            'litecoin': 'LTC-USD',
            'chainlink': 'LINK-USD',
            'polkadot': 'DOT-USD',
            'dogecoin': 'DOGE-USD'
        }
    
    def get_crypto_data(self, crypto_name, period='1y'):
        """Getting realtime data from Yahoo Finance"""
        symbol = self.crypto_symbols.get(crypto_name)
        if not symbol:
            return None
            
        try:
            ticker = yf.Ticker(symbol) # It is used to hit Yahoo Finance API
            data = ticker.history(period=period) # this function is used to get the OHLCV data
            
            if data.empty:
                return None
                
            # Calculating basic features
            data['Daily_Return'] = data['Close'].pct_change()
            data['Price_Change'] = data['Close'].diff()
            data['Volume_Change'] = data['Volume'].pct_change()
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {crypto_name}: {e}")
            return None
    
    def get_multiple_cryptos(self, crypto_list, period='6mo'):
        """Get data for multiple cryptocurrencies"""
        data_dict = {}
        for crypto in crypto_list:
            data = self.get_crypto_data(crypto, period)
            if data is not None:
                data_dict[crypto] = data
        return data_dict