import os
import streamlit as st
import google.generativeai as genai
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import requests
from dotenv import load_dotenv
import matplotlib.dates as mdates
import pandas as pd
from textblob import TextBlob

# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
finnhub_api_key = os.getenv("FINNHUB_API_KEY")

# Configure the Google GenerativeAI
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-pro')

def get_exchange_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        currency = info.get('currency', 'USD')
        exchange = info.get('exchange', 'Unknown')
        return currency, exchange
    except:
        return 'USD', 'Unknown'

def format_currency(amount, currency, target_currency='USD'):
    if amount == 'N/A':
        return 'N/A'

    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{currency}"
        response = requests.get(url)
        rates = response.json()['rates']

        if currency != target_currency:
            converted = amount * rates[target_currency]
            return f"{currency} {amount:.2f} ({target_currency} {converted:.2f})"
        return f"{currency} {amount:.2f}"
    except:
        return f"{currency} {amount:.2f}"

def get_stock_data(symbol, period='1mo'):
    stock = yf.Ticker(symbol)
    history = stock.history(period=period)

    info = stock.info
    currency, exchange = get_exchange_info(symbol)

    # Enhanced financial metrics
    metrics = {
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'forward_pe': info.get('forwardPE', 'N/A'),
        'market_cap': format_currency(info.get('marketCap', 'N/A')/1e9, currency) if info.get('marketCap') else 'N/A',
        'dividend_yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
        'beta': info.get('beta', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'currency': currency,
        'exchange': exchange,
        '52_week_high': format_currency(info.get('fiftyTwoWeekHigh', 'N/A'), currency),
        '52_week_low': format_currency(info.get('fiftyTwoWeekLow', 'N/A'), currency)
    }

    return {'history': history, 'metrics': metrics}

def analyze_news_impact(news_data, stock_data):
    try:
        if not news_data or len(news_data) == 0:
            return "No news data available for analysis."

        sentiments = []
        for article in news_data:
            if isinstance(article, dict):
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():  # Only analyze if there's text content
                    analysis = TextBlob(text)
                    sentiments.append(analysis.sentiment.polarity)

        if not sentiments:
            return "No sentiment data available for analysis."

        # Added check for empty sentiments list
        if len(sentiments) == 0:
            return "Unable to analyze sentiment due to insufficient data."

        avg_sentiment = sum(sentiments) / len(sentiments)

        # Calculate recent price change with error handling
        try:
            if len(stock_data['history']) > 1:
                recent_price_change = stock_data['history']['Close'].pct_change().iloc[-1] * 100
            else:
                recent_price_change = 0
        except (KeyError, IndexError):
            recent_price_change = 0

        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"

        impact_analysis = f"""
        News Sentiment Analysis:
        - Overall Sentiment Score: {avg_sentiment:.2f} (-1 to 1 scale)
        - Sentiment Classification: {sentiment_label}
        - Recent Price Change: {recent_price_change:.2f}%

        Analysis:
        The news sentiment is {sentiment_label.lower()} with a score of {avg_sentiment:.2f},
        while the stock has moved {recent_price_change:.2f}% recently.
        This suggests {'alignment' if (avg_sentiment > 0 and recent_price_change > 0) or
        (avg_sentiment < 0 and recent_price_change < 0) else 'divergence'}
        between news sentiment and price action.
        """

        return impact_analysis
    except Exception as e:
        st.error(f"Error in news impact analysis: {str(e)}")
        return "Unable to analyze news impact due to insufficient data."

def get_enhanced_news_sentiment(symbol):
    """Get news and sentiment analysis from multiple sources"""
    try:
        # Add error handling for API keys
        if not news_api_key or not finnhub_api_key:
            return "API keys not properly configured.", []

        news_sources = [
             f'https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}&language=en&sortBy=publishedAt&pageSize=10',
            #  f'https://finnhub.io/api/v1/company-news?symbol={symbol}&token={finnhub_api_key}'
        ]

        all_articles = []
        for url in news_sources:
            try:
                response = requests.get(url, timeout=10)  # Added timeout
                response.raise_for_status()  # Check for HTTP errors

                if 'newsapi.org' in url:
                    articles = response.json().get('articles', [])
                else:
                    articles = response.json() if response.json() else []
                all_articles.extend(articles)
            except requests.exceptions.RequestException as e:
                st.warning(f"Error fetching news from one source: {str(e)}")
                continue

        if not all_articles:
            return "No recent news found.", []

        # Format articles consistently
        formatted_articles = []
        for article in all_articles:
            if isinstance(article, dict):
                formatted_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'url': article.get('url', '')
                }
                # Only add articles with titles
                if formatted_article['title'].strip():
                    formatted_articles.append(formatted_article)

        # Sort and limit articles
        formatted_articles = sorted(
            formatted_articles,
            key=lambda x: x.get('publishedAt', ''),
            reverse=True
        )[:10]

        if not formatted_articles:
            return "No valid news articles found.", []

        # Generate analysis with error handling
        try:
            prompt = f"""
            Analyze these recent news headlines about {symbol}:
            {[article['title'] for article in formatted_articles]}

            Provide:
            1. Overall market sentiment (positive, negative, or neutral)
            2. Potential impact on stock price
            3. Key themes or patterns in the news
            4. Any significant announcements or events
            """

            response = model.generate_content(prompt)
            return response.text, formatted_articles
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return "Error generating sentiment analysis.", formatted_articles

    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return "Error fetching news data.", []

def plot_enhanced_stock_data(data, symbol):
    try:
        # Calculate technical indicators
        bb = BollingerBands(data['Close'])
        macd = MACD(data['Close'])
        stoch = StochasticOscillator(data['High'], data['Low'], data['Close'])

        data['BB_high'] = bb.bollinger_hband()
        data['BB_low'] = bb.bollinger_lband()
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['Stoch_k'] = stoch.stoch()
        data['Stoch_d'] = stoch.stoch_signal()

        # Create subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        # Price and Bollinger Bands
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(data.index, data['Close'], label='Close', color='blue')
        ax1.plot(data.index, data['BB_high'], '--', label='BB Upper', color='gray', alpha=0.5)
        ax1.plot(data.index, data['BB_low'], '--', label='BB Lower', color='gray', alpha=0.5)
        ax1.fill_between(data.index, data['BB_high'], data['BB_low'], alpha=0.1, color='gray')
        ax1.set_title(f'{symbol} Technical Analysis', fontsize=16)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # MACD
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
        ax2.plot(data.index, data['MACD_signal'], label='Signal', color='red')
        ax2.bar(data.index, data['MACD'] - data['MACD_signal'], alpha=0.3, color='gray')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        # Stochastic Oscillator
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(data.index, data['Stoch_k'], label='%K', color='blue')
        ax3.plot(data.index, data['Stoch_d'], label='%D', color='red')
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.3)
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.3)
        ax3.legend(loc='upper left')
        ax3.grid(True)

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting stock data: {e}")
        return None

def generate_prediction(symbol, history, pe_ratio, market_cap, dividend_yield):
    try:
        # Placeholder for AI prediction logic
        prompt = f"""
        Generate a prediction for the stock {symbol} based on the following data:
        - Historical price data: {history}
        - P/E Ratio: {pe_ratio}
        - Market Cap: {market_cap}
        - Dividend Yield: {dividend_yield}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating prediction: {e}"

# Streamlit UI
st.title("📈 StockIQ - Universal Stock Analysis Platform")
st.markdown("""
This enhanced version supports stocks from any exchange worldwide and provides comprehensive analysis including:
- Technical indicators and advanced charting
- Multi-source news sentiment analysis
- Price impact correlation
- Global currency conversion
""")

# Add session state for error handling
if 'error' not in st.session_state:
    st.session_state.error = None

user_input = st.text_input("Enter a stock symbol (e.g., AAPL, VOW.DE, 6758.T):",
                          help="You can enter symbols from any exchange. Examples: AAPL (US), VOW.DE (Germany), 6758.T (Tokyo)")

if user_input:
    symbol = user_input.upper()

    try:
        with st.spinner('Loading data...'):
            stock_data = get_stock_data(symbol)

        if not stock_data['history'].empty:  # Check if we got valid data
            tabs = st.tabs(["📊 Technical Analysis", "📰 News & Sentiment", "📈 Predictions", "ℹ️ Company Info"])

            # Technical Analysis Tab
            with tabs[0]:
                period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y"])
                stock_data = get_stock_data(symbol, period)
                fig = plot_enhanced_stock_data(stock_data['history'], symbol)
                if fig:
                    st.pyplot(fig)

                with st.expander("Technical Indicators Guide"):
                    st.write("""
                    - **Bollinger Bands**: Volatility indicator showing potential overbought/oversold conditions
                    - **MACD**: Trend-following momentum indicator showing potential trend changes
                    - **Stochastic Oscillator**: Momentum indicator comparing closing price to price range
                    """)

            # News & Sentiment Tab
            with tabs[1]:
                sentiment_analysis, news_articles = get_enhanced_news_sentiment(symbol)
                impact_analysis = analyze_news_impact(news_articles, stock_data)

                st.subheader("News Impact Analysis")
                st.write(impact_analysis)

                st.subheader("Detailed Sentiment Analysis")
                st.write(sentiment_analysis)

                with st.expander("Recent News"):
                    if news_articles:
                        for article in news_articles[:5]:
                            st.markdown(f"""
                            **{article['title']}**
                            - {article.get('description', 'No description available')}
                            - Published: {article.get('publishedAt', 'N/A')}
                            - [Read more]({article.get('url', '#')})
                            ---
                            """)
                    else:
                        st.info("No recent news articles found.")

            # Predictions Tab
            with tabs[2]:
                st.subheader("AI-Powered Market Analysis")
                prediction = generate_prediction(symbol, stock_data['history'], stock_data['metrics']['pe_ratio'],
                                                 stock_data['metrics']['market_cap'], stock_data['metrics']['dividend_yield'])
                st.write(prediction)

                st.info("Note: Predictions are based on historical data and current market conditions. Always perform your own due diligence before making investment decisions.")

            # Company Info Tab
            with tabs[3]:
                metrics = stock_data['metrics']

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Exchange", metrics['exchange'])
                    st.metric("Sector", metrics['sector'])
                    st.metric("P/E Ratio", metrics['pe_ratio'])
                    st.metric("Market Cap", metrics['market_cap'])

                with col2:
                    st.metric("Currency", metrics['currency'])
                    st.metric("Industry", metrics['industry'])
                    st.metric("Dividend Yield", metrics['dividend_yield'])
                    st.metric("Beta", metrics['beta'])

                with st.expander("52-Week Trading Range"):
                    st.write(f"High: {metrics['52_week_high']}")
                    st.write(f"Low: {metrics['52_week_low']}")
        else:
            st.error(f"No data found for symbol: {symbol}")

    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        st.session_state.error = str(e)
