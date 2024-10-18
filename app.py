!pip install yfinance tensorflow scikit-learn streamlit

import yfinance as yf
import pandas as pd

# Download stock data (example: Reliance Industries)
def get_stock_data(ticker):
    stock_data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
    return stock_data

# Fetch the stock data for Reliance
stock_data = get_stock_data("RELIANCE.NS")

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))

# Create sequences of 60 past days' prices to predict the next day's price
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

# Reshape X for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Build the LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, batch_size=64, epochs=10)

def predict_future_price(model, data, time_step, scaler):
    last_days_data = data[-time_step:]
    scaled_last_days_data = scaler.transform(last_days_data.reshape(-1,1))

    X_test = []
    X_test.append(scaled_last_days_data)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

predicted_price = predict_future_price(model, stock_data['Close'].values[-time_step:], time_step, scaler)
print(f"Predicted price: {predicted_price}")

import streamlit as st

st.title("Indian Stock Price Prediction")

# User input for stock ticker
ticker = st.text_input("Enter the stock ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

if st.button('Predict'):
    data = get_stock_data(ticker)
    predicted_price = predict_future_price(model, data['Close'].values[-time_step:], time_step, scaler)
    st.write(f"Predicted price for {ticker}: {predicted_price[0][0]}")