import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
    
    def fit(self, series, order=(1,1,1)):
        ''' Fitting ARIMA model to the series '''
        try:
            self.model = ARIMA(series, order=order)
            self.fitted_model = self.model.fit()
            return True
        except Exception as e:
            print(f"ARIMA fitting error: {e}")
            return False
    
    def forecast(self, steps=30):
        """Generate forecasts"""
        if self.fitted_model:
            try:
                forecast = self.fitted_model.forecast(steps=steps)
                return forecast
            except Exception as e:
                print(f"ARIMA forecasting error: {e}")
                return None
        return None
    
    def find_best_parameters(self, series):
        """parameter optimization"""
        best_aic = np.inf
        best_order = (1,1,1)
        
        orders_to_try = [(1,1,1), (2,1,2), (1,1,0), (0,1,1)]
        
        for order in orders_to_try:
            try:
                model = ARIMA(series, order=order)
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = order
            except:
                continue
        
        return best_order, best_aic