import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
from datetime import datetime
from textblob import TextBlob
import requests

# Set your NewsAPI key
newsapi_key = "705b2d1692b7464f861732a259cbd380"

# Streamlit app title
st.title("Stock Predictor with Sentiment Analysis")

# Input stock symbol
ticker = st.text_input("Enter Stock Symbol", "AAPL")

# Fetch stock data
stock_data = yf.download(ticker, start="2010-01-01", end="2024-12-14")

if stock_data.empty:
    st.error("No data found for the ticker symbol.")
    st.stop()

# Display the stock data
st.subheader("Stock Data")
st.write(stock_data.tail())

# Prepare data for prediction
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].map(lambda x: x.toordinal())  # Convert to ordinal dates
X = stock_data[['Date']].values  # Convert to 2D array
y = stock_data['Close'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Short-term predictions
predictions = model.predict(X_test).flatten()  # Ensure predictions are 1-dimensional
prediction_dates = X_test.flatten()  # Ensure dates are 1-dimensional

# Convert ordinal dates to datetime
prediction_dates = [datetime.fromordinal(int(date)) for date in prediction_dates]

# Create prediction DataFrame
prediction_df = pd.DataFrame({
    'Date': prediction_dates,
    'Predicted Close': predictions
})

# Long-term predictions (1 year ahead)
future_dates = [X[-1][0] + i for i in range(1, 366)]
future_dates_datetime = [datetime.fromordinal(int(date)) for date in future_dates]
long_term_predictions = model.predict(np.array(future_dates).reshape(-1, 1)).flatten()  # Flatten predictions

# Long-term DataFrame
long_term_df = pd.DataFrame({
    'Date': future_dates_datetime,
    'Predicted Close': long_term_predictions
})

# Fetch live stock news using NewsAPI
def fetch_stock_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={newsapi_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    headlines = [article["title"] for article in articles[:5]]  # Get top 5 headlines
    return headlines

# Sentiment analysis function
def analyze_sentiment(news_headlines):
    sentiment_scores = []
    for headline in news_headlines:
        analysis = TextBlob(headline)
        sentiment_scores.append(analysis.sentiment.polarity)
    average_score = np.mean(sentiment_scores) if sentiment_scores else 0
    return average_score, sentiment_scores

news_headlines = fetch_stock_news(ticker)
sentiment_score, sentiment_scores = analyze_sentiment(news_headlines)

# Display news and sentiment
st.subheader(f"Live News for {ticker}")
for headline in news_headlines:
    st.write(f"- {headline}")

st.subheader("Sentiment Analysis")
st.write(f"Average Sentiment Score: {sentiment_score}")
if sentiment_score > 0.2:
    st.success("Strong Buy")
elif 0.1 < sentiment_score <= 0.2:
    st.success("Buy")
elif -0.1 <= sentiment_score <= 0.1:
    st.info("Neutral")
elif -0.2 <= sentiment_score < -0.1:
    st.error("Sell")
else:
    st.error("Strong Sell")

# Function to classify price direction (Buy/Sell)
def classify_price_direction(prev_price, curr_price):
    if curr_price > prev_price:
        return "Buy", "green"
    elif curr_price < prev_price:
        return "Sell", "red"
    else:
        return "Neutral", "gray"

# Plotting predictions and sentiment
fig, ax = plt.subplots(figsize=(12, 6))

# Plot real prices
ax.plot(stock_data.index, stock_data['Close'], label="Real Prices", color="blue")

# Plot short-term predictions
ax.plot(prediction_df['Date'], prediction_df['Predicted Close'], label="Short-term Predictions", color="red")

# Plot long-term predictions
ax.plot(long_term_df['Date'], long_term_df['Predicted Close'], label="Long-term Predictions", color="green")

ax.set_title(f"Stock Predictions for {ticker}")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Display prediction DataFrame
st.subheader("Short-term Predictions")
for i, row in prediction_df.iterrows():
    action, color = classify_price_direction(prediction_df['Predicted Close'].iloc[i-1] if i > 0 else row['Predicted Close'], row['Predicted Close'])
    st.write(f"{row['Date'].date()}: Predicted Close: {row['Predicted Close']:.2f} - {action}", f"Color: {color}")
    st.markdown(f"<p style='color:{color};'>{action}</p>", unsafe_allow_html=True)

st.subheader("Long-term Predictions")
for i, row in long_term_df.iterrows():
    action, color = classify_price_direction(long_term_df['Predicted Close'].iloc[i-1] if i > 0 else row['Predicted Close'], row['Predicted Close'])
    st.write(f"{row['Date'].date()}: Predicted Close: {row['Predicted Close']:.2f} - {action}", f"Color: {color}")
    st.markdown(f"<p style='color:{color};'>{action}</p>", unsafe_allow_html=True)