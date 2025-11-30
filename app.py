# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_collector import CryptoDataCollector
from data_preprocessing import DataProcessor
from ARIMA_model import ARIMAModel
from LSTM_model import LSTMModel
from sentiment_analyzer import SentimentAnalyzer
from volatility_analyzer import VolatilityAnalyzer

class CryptoAnalysisApp:
    def __init__(self):
        self.data_collector = CryptoDataCollector()
        self.data_processor = DataProcessor()
        self.arima_model = ARIMAModel()
        self.lstm_model = LSTMModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        
    def run(self):
        # Page configuration
        st.set_page_config(
            page_title="Crypto Analysis Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🚀 Complete Crypto Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_main_content()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Cryptocurrency selection
            crypto_options = {
                'bitcoin': 'Bitcoin (BTC)',
                'ethereum': 'Ethereum (ETH)', 
                'cardano': 'Cardano (ADA)',
                'ripple': 'Ripple (XRP)',
                'litecoin': 'Litecoin (LTC)',
                'chainlink': 'Chainlink (LINK)',
                'polkadot': 'Polkadot (DOT)',
                'dogecoin': 'Dogecoin (DOGE)'
            }
            
            self.selected_crypto = st.selectbox(
                "Select Cryptocurrency",
                list(crypto_options.keys()),
                format_func=lambda x: crypto_options[x]
            )
            
            # Time period
            self.period = st.selectbox(
                "Time Period",
                ['1mo', '3mo', '6mo', '1y', '2y'],
                index=3
            )
            
            # Analysis type
            self.analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "📊 Market Overview",
                    "📈 Technical Analysis", 
                    "🔮 Price Forecasting",
                    "⚡ Volatility Analysis",
                    "😊 Sentiment Analysis",
                    "🔄 Multi-Asset Comparison"
                ]
            )
            
            # Refresh button
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Quick Stats")
            self.render_quick_stats()
    
    def render_quick_stats(self):
        """Render quick statistics in sidebar"""
        if hasattr(self, 'current_data') and self.current_data is not None:
            current_price = self.current_data['Close'].iloc[-1]
            prev_price = self.current_data['Close'].iloc[-2] if len(self.current_data) > 1 else current_price
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            st.metric(
                "Current Price", 
                f"${current_price:,.2f}",
                f"{price_change_pct:+.2f}%"
            )
            
            # RSI indicator
            current_rsi = self.current_data['RSI'].iloc[-1]
            rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            st.metric("RSI", f"{current_rsi:.1f}", rsi_status)
    
    def render_main_content(self):
        """Render main content based on selection"""
        # Load data
        with st.spinner(f"Loading {self.selected_crypto} data..."):
            self.load_data()
        
        if self.current_data is None:
            st.error("❌ Failed to load data. Please try again.")
            return
        
        # Route to appropriate analysis
        if self.analysis_type == "📊 Market Overview":
            self.render_market_overview()
        elif self.analysis_type == "📈 Technical Analysis":
            self.render_technical_analysis()
        elif self.analysis_type == "🔮 Price Forecasting":
            self.render_forecasting()
        elif self.analysis_type == "⚡ Volatility Analysis":
            self.render_volatility_analysis()
        elif self.analysis_type == "😊 Sentiment Analysis":
            self.render_sentiment_analysis()
        elif self.analysis_type == "🔄 Multi-Asset Comparison":
            self.render_multi_asset_analysis()
    
    def load_data(self):
        """Load and process data"""
        try:
            raw_data = self.data_collector.get_crypto_data(
                self.selected_crypto, 
                self.period
            )
            
            if raw_data is not None:
                # Process data with technical indicators
                self.current_data = self.data_processor.calculate_technical_indicators(raw_data)
                st.session_state.current_data = self.current_data
            else:
                st.error("No data returned from API")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    def render_market_overview(self):
        """Render market overview dashboard"""
        st.header("📊 Market Overview")
        
        # Performance metrics
        perf_metrics = self.volatility_analyzer.performance_metrics(self.current_data['Close'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{perf_metrics['total_return_pct']:.1f}%")
        
        with col2:
            st.metric("Annualized Return", f"{perf_metrics['annualized_return_pct']:.1f}%")
        
        with col3:
            st.metric("Max Drawdown", f"{perf_metrics['max_drawdown_pct']:.1f}%")
        
        with col4:
            st.metric("Win Rate", f"{perf_metrics['win_rate']:.1f}%")
        
        # Price chart with indicators
        st.subheader("Price Chart with Technical Indicators")
        fig = self.create_main_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Market insights
        st.subheader("🎯 Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trend analysis
            trend = self.data_processor.detect_trend(self.current_data)
            st.info(f"**Trend Analysis**: {trend}")
            
            # Support and Resistance
            support = self.current_data['Support'].iloc[-1]
            resistance = self.current_data['Resistance'].iloc[-1]
            current_price = self.current_data['Close'].iloc[-1]
            
            st.write("**Key Levels:**")
            st.write(f"- Support: ${support:,.2f}")
            st.write(f"- Resistance: ${resistance:,.2f}")
            st.write(f"- Current: ${current_price:,.2f}")
        
        with col2:
            # Trading signals
            signals = self.data_processor.generate_trading_signals(self.current_data)
            if signals:
                st.success("**Trading Signals:**")
                for signal in signals:
                    st.write(f"{signal[2]} **{signal[0]}**: {signal[1]}")
            else:
                st.info("No strong trading signals detected")
    
    def create_main_chart(self):
        """Create main price chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{self.selected_crypto.title()} Price Chart',
                'RSI',
                'MACD'
            ),
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=self.current_data.index,
                open=self.current_data['Open'],
                high=self.current_data['High'],
                low=self.current_data['Low'],
                close=self.current_data['Close'],
                name='Price'
            ), row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=self.current_data.index, y=self.current_data['SMA_20'], 
                      name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.current_data.index, y=self.current_data['SMA_50'], 
                      name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=self.current_data.index, y=self.current_data['RSI'], 
                      name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=self.current_data.index, y=self.current_data['MACD'], 
                      name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.current_data.index, y=self.current_data['MACD_Signal'], 
                      name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(height=700, showlegend=True, 
                         xaxis_rangeslider_visible=False)
        return fig
    
    def render_technical_analysis(self):
        """Render technical analysis"""
        st.header("📈 Technical Analysis")
        
        # Individual indicator analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RSI Analysis")
            current_rsi = self.current_data['RSI'].iloc[-1]
            
            if current_rsi > 70:
                st.error(f"RSI: {current_rsi:.1f} - Overbought (Sell Signal)")
            elif current_rsi < 30:
                st.success(f"RSI: {current_rsi:.1f} - Oversold (Buy Signal)")
            else:
                st.info(f"RSI: {current_rsi:.1f} - Neutral")
            
            # RSI Chart
            fig_rsi = px.line(self.current_data, x=self.current_data.index, y='RSI',
                             title='RSI Indicator')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            st.subheader("MACD Analysis")
            macd = self.current_data['MACD'].iloc[-1]
            signal = self.current_data['MACD_Signal'].iloc[-1]
            
            if macd > signal:
                st.success("MACD above Signal Line - Bullish")
            else:
                st.warning("MACD below Signal Line - Bearish")
            
            # MACD Chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=self.current_data.index, y=self.current_data['MACD'],
                                         name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=self.current_data.index, y=self.current_data['MACD_Signal'],
                                         name='Signal', line=dict(color='red')))
            fig_macd.update_layout(title='MACD Indicator')
            st.plotly_chart(fig_macd, use_container_width=True)
    
    def render_forecasting(self):
        """Render price forecasting with all models"""
        st.header("🔮 Price Forecasting")
        
        model_choice = st.selectbox(
            "Select Forecasting Model",
            ["ARIMA", "LSTM", "Linear Regression"]
        )
        
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                if model_choice == "ARIMA":
                    self.render_arima_forecast(forecast_days)
                elif model_choice == "LSTM":
                    self.render_lstm_forecast(forecast_days)
                else:
                    self.render_linear_forecast(forecast_days)
    
    def render_arima_forecast(self, forecast_days):
        """Render ARIMA forecasting"""
        series = self.current_data['Close'].dropna()
        
        # Find best parameters
        best_order, best_aic = self.arima_model.find_best_parameters(series)
        st.write(f"Best ARIMA Order: {best_order}, AIC: {best_aic:.2f}")
        
        # Fit model
        if self.arima_model.fit(series, best_order):
            forecast = self.arima_model.forecast(forecast_days)
            
            if forecast is not None:
                self.plot_forecast_results(series, forecast, "ARIMA")
            else:
                st.error("ARIMA forecast failed")
        else:
            st.error("ARIMA model fitting failed")
    
    def render_lstm_forecast(self, forecast_days):
        """Render LSTM forecasting"""
        series = self.current_data['Close'].dropna()
        
        predictions = self.lstm_model.train_and_predict(series, forecast_days)
        
        if predictions is not None:
            self.plot_forecast_results(series, predictions, "LSTM")
        else:
            st.error("LSTM forecast failed - try with more data")
    
    def render_linear_forecast(self, forecast_days):
        """Render simple linear regression forecast"""
        series = self.current_data['Close'].dropna()
        
        # Simple linear regression
        X = np.array(range(len(series))).reshape(-1, 1)
        y = series.values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.array(range(len(series), len(series) + forecast_days)).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        self.plot_forecast_results(series, predictions, "Linear Regression")
    
    def plot_forecast_results(self, historical, forecast, model_name):
        """Plot forecast results"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index, 
            y=historical.values,
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        future_dates = pd.date_range(
            start=historical.index[-1] + timedelta(days=1),
            periods=len(forecast)
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast,
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"{model_name} Forecast for {self.selected_crypto.title()}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        current_price = historical.iloc[-1]
        predicted_price = forecast[-1]
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        st.subheader("Forecast Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}")
        with col3:
            st.metric("Expected Change", f"{change_pct:+.1f}%")
    
    def render_volatility_analysis(self):
        """Render volatility analysis"""
        st.header("⚡ Volatility Analysis")
        
        returns = self.current_data['Daily_Return'].dropna()
        vol_metrics = self.volatility_analyzer.calculate_volatility_metrics(returns)
        risk_level, risk_color = self.volatility_analyzer.calculate_risk_level(returns)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily Volatility", f"{vol_metrics['daily_volatility']*100:.2f}%")
        
        with col2:
            st.metric("Annual Volatility", f"{vol_metrics['annual_volatility']*100:.2f}%")
        
        with col3:
            st.metric("Max Daily Gain", f"{vol_metrics['max_daily_gain']*100:.2f}%")
        
        with col4:
            st.metric("Max Daily Loss", f"{vol_metrics['max_daily_loss']*100:.2f}%")
        
        # Risk assessment
        st.subheader("📊 Risk Assessment")
        st.markdown(f"<h3 style='color: {risk_color}'>Risk Level: {risk_level}</h3>", 
                   unsafe_allow_html=True)
        
        # VaR metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Value at Risk (95%)", f"{vol_metrics['var_95']*100:.2f}%")
        with col2:
            st.metric("Conditional VaR", f"{vol_metrics['cvar_95']*100:.2f}%")
        
        # Volatility chart
        st.subheader("Volatility Over Time")
        self.current_data['Rolling_Volatility'] = returns.rolling(20).std()
        
        fig_vol = px.line(self.current_data, x=self.current_data.index, y='Rolling_Volatility',
                         title='20-Day Rolling Volatility')
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Returns distribution
        st.subheader("Returns Distribution")
        fig_hist = px.histogram(self.current_data, x='Daily_Return', 
                               nbins=50, title='Distribution of Daily Returns')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis"""
        st.header("😊 Sentiment Analysis")
        
        st.info("""
        **Sentiment Analysis** uses NLP (Natural Language Processing) to analyze 
        how people feel about cryptocurrencies based on news and social media.
        """)
        
        # Analyze sentiment
        sentiment_data = self.sentiment_analyzer.analyze_news_sentiment(self.selected_crypto)
        market_sentiment = self.sentiment_analyzer.get_market_sentiment_summary(sentiment_data)
        
        # Market sentiment
        st.subheader("Market Sentiment")
        st.success(market_sentiment)
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = sentiment_data['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="News Sentiment Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Metrics")
            avg_polarity = sentiment_data['polarity'].mean()
            avg_subjectivity = sentiment_data['subjectivity'].mean()
            
            st.metric("Average Polarity", f"{avg_polarity:.3f}")
            st.metric("Average Subjectivity", f"{avg_subjectivity:.3f}")
            st.metric("Articles Analyzed", len(sentiment_data))
        
        # News headlines with sentiment
        st.subheader("Recent News Analysis")
        for idx, row in sentiment_data.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    sentiment_emoji = "😊" if row['sentiment'] == 'positive' else "😞" if row['sentiment'] == 'negative' else "😐"
                    st.write(f"**{sentiment_emoji} {row['sentiment'].title()}**")
                    st.write(f"Score: {row['score']:.3f}")
                with col2:
                    st.write(row['headline'])
            st.divider()
    
    def render_multi_asset_analysis(self):
        """Render multi-asset comparison"""
        st.header("🔄 Multi-Asset Comparison")
        
        # Select cryptocurrencies to compare
        compare_cryptos = st.multiselect(
            "Select cryptocurrencies to compare",
            ['bitcoin', 'ethereum', 'cardano', 'ripple', 'litecoin'],
            default=['bitcoin', 'ethereum']
        )
        
        if len(compare_cryptos) < 2:
            st.warning("Please select at least 2 cryptocurrencies")
            return
        
        # Load comparison data
        with st.spinner("Loading comparison data..."):
            crypto_data = self.data_collector.get_multiple_cryptos(compare_cryptos, '6mo')
        
        if len(crypto_data) < 2:
            st.error("Insufficient data for comparison")
            return
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        closes_df = pd.DataFrame()
        for crypto, data in crypto_data.items():
            closes_df[crypto] = data['Close']
        
        correlation_matrix = closes_df.corr()
        
        fig_corr = px.imshow(correlation_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title='Cryptocurrency Correlation Matrix')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        fig_compare = go.Figure()
        for crypto, data in crypto_data.items():
            # Normalize prices to percentage
            normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
            fig_compare.add_trace(go.Scatter(
                x=data.index, y=normalized_prices,
                name=crypto.title()
            ))
        
        fig_compare.update_layout(
            title='Normalized Price Comparison (Base = 100)',
            yaxis_title='Normalized Price (%)'
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Performance table
        st.subheader("Performance Summary")
        performance_data = []
        
        for crypto, data in crypto_data.items():
            perf_metrics = self.volatility_analyzer.performance_metrics(data['Close'])
            
            performance_data.append({
                'Cryptocurrency': crypto.title(),
                'Total Return (%)': f"{perf_metrics['total_return_pct']:.2f}%",
                'Volatility (%)': f"{perf_metrics['volatility_pct']:.2f}%",
                'Max Drawdown (%)': f"{perf_metrics['max_drawdown_pct']:.2f}%",
                'Win Rate (%)': f"{perf_metrics['win_rate']:.2f}%"
            })
        
        st.table(pd.DataFrame(performance_data))

def main():
    """Main application entry point"""
    app = CryptoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()