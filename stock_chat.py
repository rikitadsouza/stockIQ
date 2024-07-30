import os
import streamlit as st
import google.generativeai as genai
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import requests
from dotenv import load_dotenv
import matplotlib.dates as mdates


# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# Configure the Google GenerativeAI
genai.configure(api_key=google_api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

def get_exchange_rate():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['rates']['INR']
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching exchange rate: {e}")
        return None

def get_stock_symbol(user_input):
    if '.NS' in user_input.upper() or '.BO' in user_input.upper():
        return user_input.upper()
    elif '.' not in user_input:
        exchange = st.selectbox("Is this an Indian or American stock?", ('IN', 'US'))
        if exchange.upper() == 'IN':
            return f"{user_input.upper()}.NS"  # Assume Indian stock on NSE
        else:
            return user_input.upper()  # Assume American stock
    else:
        return user_input.upper()

def display_price(price, symbol):
    exchange_rate = get_exchange_rate()
    if exchange_rate:
        if '.NS' in symbol or '.BO' in symbol:
            usd_price = price / exchange_rate
            return f"₹{price:.2f} (${usd_price:.2f})"
        else:
            inr_price = price * exchange_rate
            return f"${price:.2f} (₹{inr_price:.2f})"
    else:
        if '.NS' in symbol or '.BO' in symbol:
            return f"₹{price:.2f}"
        else:
            return f"${price:.2f}"

def get_stock_data(symbol, period='1mo'):
    stock = yf.Ticker(symbol)
    history = stock.history(period=period)
    
    info = stock.info
    pe_ratio = info.get('trailingPE', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    if market_cap != 'N/A':
        if '.NS' in symbol or '.BO' in symbol:
            market_cap = f"₹{market_cap / 1e7:.2f} Cr"
        else:
            market_cap = f"${market_cap / 1e9:.2f}B"
    dividend_yield = info.get('dividendYield', 'N/A')
    if dividend_yield != 'N/A':
        dividend_yield = f"{dividend_yield * 100:.2f}%"
    
    result = {
        'history': history,
        'pe_ratio': pe_ratio,
        'market_cap': market_cap,
        'dividend_yield': dividend_yield
    }
    return result

def get_news_sentiment(symbol):
    try:
        url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}&language=en&sortBy=publishedAt&pageSize=5'
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        
        if not articles:
            return "No recent news found."
        
        titles = [article['title'] for article in articles]
        
        prompt = f"""
        Analyze the sentiment of these recent news headlines about {symbol}:
        {titles}
        
        Provide a brief summary of the overall sentiment and how it might affect the stock price.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return "Failed to fetch news sentiment."


def plot_stock_data(data, symbol):
    try:
        sma = SMAIndicator(data['Close'], window=20)
        data['SMA'] = sma.sma_indicator()
        
        rsi = RSIIndicator(data['Close'])
        data['RSI'] = rsi.rsi()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        ax1.plot(data.index, data['Close'], label='Close', color='b')
        ax1.plot(data.index, data['SMA'], label='20-day SMA', color='orange')
        ax1.set_title(f'{symbol} Stock Price and Indicators', fontsize=16)
        ax1.set_ylabel('Price', fontsize=14)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True)
        
        ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax2.set_ylabel('RSI', fontsize=14)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True)

        # Set the date format for x-axis
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)

        fig.autofmt_xdate()  # Auto format date labels
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        st.pyplot(fig)  # Display the plot in the Streamlit app
    except Exception as e:
        st.error(f"Error plotting stock data: {e}")  # Display an error message in the Streamlit app

def generate_prediction(symbol, data, pe_ratio, market_cap, dividend_yield):
    latest_price = data['Close'].iloc[-1]
    week_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100)
    period_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
    
    prompt = f"""
    Analyze the following stock data for {symbol} and provide a prediction for the next week:

    Latest price: {display_price(latest_price, symbol)}
    1-week change: {week_change:.2f}%
    Period change: {period_change:.2f}%
    P/E Ratio: {pe_ratio}
    Market Cap: {market_cap}
    Dividend Yield: {dividend_yield}

    Consider market trends, recent news, historical performance, and these financial metrics. Provide a brief explanation for your prediction.
    """
    
    response = model.generate_content(prompt)
    return response.text

def explain_financial_metric(metric):
    prompt = f"""
    Explain the following financial metric in simple terms: {metric}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("StockIQ - Stock Prediction Chatbot")

user_input = st.text_input("Enter a stock symbol: ")

if user_input:
    if user_input.upper() != 'QUIT':
        symbol = get_stock_symbol(user_input)
        
        choice = st.selectbox(
            "Choose an option:",
            ["Get stock prediction", "View historical data", "Understand financial metrics", "Get news sentiment analysis", "Choose another stock"]
        )

        if choice == "Get stock prediction":
            period = st.selectbox("Select the time period", ["1mo", "3mo", "6mo", "1y"])
            if period:
                try:
                    stock_data = get_stock_data(symbol, period)
                    data = stock_data['history']
                    plot_stock_data(data, symbol)
                    prediction = generate_prediction(symbol, data, stock_data['pe_ratio'], stock_data['market_cap'], stock_data['dividend_yield'])
                    
                    st.write(f"\nStock data for {symbol}:")
                    st.write(f"Latest price: {display_price(data['Close'].iloc[-1], symbol)}")
                    st.write(f"1-week change: {((data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100):.2f}%")
                    st.write(f"Period change: {((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100):.2f}%")
                    
                    st.write("\nPrediction:")
                    st.write(prediction)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
        elif choice == "View historical data":
            period = st.selectbox("Select the time period", ["1mo", "3mo", "6mo", "1y"])
            if period:
                try:
                    stock_data = get_stock_data(symbol, period)
                    st.write(f"\nHistorical data for {symbol}:")
                    st.write(stock_data['history'])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
        elif choice == "Understand financial metrics":
            metric = st.text_input("Enter the financial metric you want to understand (e.g., P/E Ratio, Market Cap, Dividend Yield): ")
            if metric:
                try:
                    explanation = explain_financial_metric(metric)
                    st.write(f"\nExplanation for {metric}:")
                    st.write(explanation)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
        elif choice == "Get news sentiment analysis":
            try:
                sentiment_analysis = get_news_sentiment(symbol)
                st.write(f"\nNews sentiment analysis for {symbol}:")
                st.write(sentiment_analysis)
            except Exception as e:
                st.error(f"An error occurred: {e}")