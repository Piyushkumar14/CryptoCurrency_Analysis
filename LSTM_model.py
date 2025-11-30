# src/lstm_model.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

class LSTMModel:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data):
        """Prepare data for LSTM"""
        # Use simple scaling
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def create_simple_lstm(self, input_shape):
        """Creating a LSTM model"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_and_predict(self, data, future_steps=30):
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if len(X) < 10:  # Not enough data
                return None
                
            # Splitting data (80-20 split)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Create and train model
            model = self.create_simple_lstm((X_train.shape[1], 1))
            
            # Simple training
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=20,
                batch_size=32,
                verbose=0
            )
            
            # Making predictions
            last_sequence = data[-self.lookback:].values.reshape(1, -1)
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            predictions = []
            current_sequence = last_sequence_scaled.flatten()
            
            for _ in range(future_steps):
                next_pred = model.predict(current_sequence.reshape(1, self.lookback, 1), verbose=0)[0, 0]
                predictions.append(next_pred)
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            st.error(f"LSTM Error: {e}")
            return None