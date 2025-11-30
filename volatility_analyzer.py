# src/volatility_analyzer.py
import pandas as pd
import numpy as np
from scipy import stats

class VolatilityAnalyzer:
    def calculate_volatility_metrics(self, returns_series):
        """Calculate comprehensive volatility metrics"""
        returns = returns_series.dropna()
        
        metrics = {
            # Basic volatility
            'daily_volatility': returns.std(),
            'weekly_volatility': returns.std() * np.sqrt(7),
            'monthly_volatility': returns.std() * np.sqrt(30),
            'annual_volatility': returns.std() * np.sqrt(365),
            
            # Extreme moves
            'max_daily_gain': returns.max(),
            'max_daily_loss': returns.min(),
            'avg_daily_move': returns.abs().mean(),
            
            # Risk metrics
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
            'var_95': returns.quantile(0.05),  # Value at Risk 95%
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),  # Conditional VaR
        }
        
        return metrics
    
    def analyze_volatility_regimes(self, prices):
        """Identify high/low volatility periods"""
        returns = prices.pct_change().dropna()
        rolling_vol = returns.rolling(20).std()
        
        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)
        
        regimes = []
        for vol in rolling_vol:
            if vol > high_vol_threshold:
                regimes.append("High Volatility")
            elif vol < low_vol_threshold:
                regimes.append("Low Volatility")
            else:
                regimes.append("Normal Volatility")
        
        return regimes
    
    def calculate_risk_level(self, returns_series):
        """Simple risk level classification"""
        daily_vol = returns_series.std()
        
        if daily_vol < 0.02:
            return "Low Risk", "green"
        elif daily_vol < 0.05:
            return "Medium Risk", "orange"
        else:
            return "High Risk", "red"
    
    def performance_metrics(self, prices):
        """Calculate performance metrics"""
        returns = prices.pct_change().dropna()
        
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annualized_return = ((1 + returns.mean()) ** 365 - 1) * 100
        max_drawdown = self.calculate_max_drawdown(prices)
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility_pct': returns.std() * 100,
            'win_rate': (returns > 0).mean() * 100
        }
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()