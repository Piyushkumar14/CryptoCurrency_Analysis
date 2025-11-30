import pandas as pd
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        # crypto-related word lists
        self.positive_words = {
            'bullish', 'moon', 'rocket', 'buy', 'growth', 'adoption', 
            'institutional', 'breakout', 'surge', 'rally', 'green',
            'profit', 'gain', 'success', 'innovation'
        }
        self.negative_words = {
            'bearish', 'crash', 'dump', 'sell', 'scam', 'regulation',
            'fud', 'warning', 'red', 'correction', 'loss', 'risk',
            'fraud', 'bubble', 'warning'
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob and custom word lists"""
        if not isinstance(text, str):
            return {'sentiment': 'neutral', 'score': 0, 'polarity': 0}
        
        # TextBlob sentiment
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Custom word-based analysis
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Combined score
        custom_score = positive_count - negative_count
        final_score = polarity + (custom_score * 0.1)  # Weight custom analysis
        
        # Determining sentiment
        if final_score > 0.1:
            sentiment = 'positive'
        elif final_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': final_score,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def get_sample_crypto_news(self):
        """Get sample cryptocurrency news for analysis"""
        sample_news = [
            "Bitcoin surges to new yearly high as institutional adoption grows",
            "Cryptocurrency market faces correction amid regulatory concerns",
            "Ethereum upgrade successfully implemented, improving network efficiency",
            "Major financial institution announces Bitcoin investment program",
            "Market analysts predict continued growth for digital assets",
            "Regulatory uncertainty causes volatility in crypto markets",
            "DeFi platforms see massive growth in user adoption",
            "Cryptocurrency mining faces environmental scrutiny",
            "NFT market reaches new milestones with record sales",
            "Central banks exploring digital currency alternatives"
        ]
        return sample_news
    
    def analyze_news_sentiment(self, crypto_name='bitcoin'):
        """Analyze sentiment from crypto news headlines"""
        news_headlines = self.get_sample_crypto_news()
        
        results = []
        for headline in news_headlines:
            sentiment = self.analyze_sentiment(headline)
            sentiment['headline'] = headline
            results.append(sentiment)
        
        return pd.DataFrame(results)
    
    def get_market_sentiment_summary(self, sentiment_data):
        """Generate market sentiment summary"""
        if sentiment_data.empty:
            return "No sentiment data available"
        
        positive_count = (sentiment_data['sentiment'] == 'positive').sum()
        negative_count = (sentiment_data['sentiment'] == 'negative').sum()
        neutral_count = (sentiment_data['sentiment'] == 'neutral').sum()
        total = len(sentiment_data)
        
        avg_polarity = sentiment_data['polarity'].mean()
        
        if positive_count > negative_count and avg_polarity > 0.1:
            return "Bullish Market Sentiment 🚀"
        elif negative_count > positive_count and avg_polarity < -0.1:
            return "Bearish Market Sentiment 🐻"
        else:
            return "Neutral Market Sentiment ⚖️"