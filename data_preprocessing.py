import pandas as pd
import numpy as np

class DataProcessor:
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volatility
        df['Volatility_20D'] = df['Daily_Return'].rolling(20).std()
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_trading_signals(self, df):
        """Generate simple trading signals"""
        signals = []
        
        # RSI signals
        current_rsi = df['RSI'].iloc[-1]
        if current_rsi > 70:
            signals.append(("RSI Overbought", "Consider Selling", "🔴", "High"))
        elif current_rsi < 30:
            signals.append(("RSI Oversold", "Consider Buying", "🟢", "High"))
        
        # MACD signals
        if (df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and 
            df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]):
            signals.append(("MACD Bullish Crossover", "Buy Signal", "🟢", "Medium"))
        
        # Moving Average signals
        if (df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] and 
            df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]):
            signals.append(("Golden Cross", "Strong Buy", "🟢", "High"))
        
        return signals
    
    def detect_trend(self, df):
        """Simple trend detection"""
        price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        if price > sma_20 > sma_50:
            return "Strong Uptrend 📈"
        elif price < sma_20 < sma_50:
            return "Strong Downtrend 📉"
        elif price > sma_20:
            return "Weak Uptrend ↗️"
        else:
            return "Weak Downtrend ↘️"