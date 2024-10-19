'''
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

st.title("Stock Dashboard")

ticker = st.sidebar.text_input("Ticker")
strat_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

data = yf.download(ticker, start=strat_date, end=end_date)

if data.empty:
    st.error("No data available for the selected ticker.")
else:
    fig = px.line(data, x=data.index, y=data["Adj Close"], title=f"{ticker} Stock Price")
    st.plotly_chart(fig)

pricing_data, fundamental_data, news, openai1, tech_indicator, lstm_prediction = st.tabs(
    ["Pricing Data", "Fundamental Data", "Top 10 News", "OpenAI ChatGPT", "Technical Analysis Dashboard", "LSTM Prediction"]
)

with pricing_data:
    if "Adj Close" in data.columns:
        data2 = data.copy()  
        data2["% Change"] = data2["Adj Close"] / data2["Adj Close"].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)  
        

        annual_return = data2["% Change"].mean() * 252 * 100
        st.write("Annual Return is ", annual_return, "%")
        stdev = np.std(data2["% Change"]) * np.sqrt(252)
        st.write("Standard Deviation is ", stdev * 100, "%")
        st.write("Risk Adjusted Return is ", annual_return / (stdev * 100))
    else:
        st.write("The 'Adj Close' column is missing in the data.")


with lstm_prediction:
    st.subheader("Stock Price Prediction using LSTM")
    if st.button("Predict Future Prices"):
       
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Adj Close"].values.reshape(-1, 1))

        
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        x_train, y_train = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer

       
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=50, batch_size=32)

        
        test_data = scaled_data[train_size - 60:]
        x_test, y_test = [], []

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            y_test.append(test_data[i, 0])

        x_test = np.array(x_test)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)  # Inverse scaling

        
        train = data[:train_size]
        valid = data[train_size:]
        valid['Predictions'] = predictions

        
        fig2 = px.line(valid, x=valid.index, y=["Adj Close", "Predictions"], title=f"{ticker} Stock Price Prediction")
        st.plotly_chart(fig2)


with fundamental_data:
    from alpha_vantage.fundamentaldata import FundamentalData
    key = "0M5NL6LVC4IF4ATX."
    fd = FundamentalData(key, output_format="Pandas")
    st.subheader("Balance Sheet")
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader("Income Statement")
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader("Cash Flow Statement")
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)


with news:
    from stocknews import StockNews
    st.header(f"News of {ticker}")
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f"News {i+1}")
        st.write(df_news["published"][i])
        st.write(df_news["title"][i])
        st.write(df_news["summary"][i])
        title_sentiment = df_news["sentiment_title"][i]
        st.write(f"Title Sentiment {title_sentiment}")
        news_sentiment = df_news["sentiment_summary"][i]
        st.write(f"News Sentiment {news_sentiment}")


import openai


openai.api_key = "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"  # Make sure to replace with your actual API key

def get_chatgpt_response(prompt):
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()


ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")

if ticker:
    buy = get_chatgpt_response(f"3 Reasons to buy {ticker} Stock")
    sell = get_chatgpt_response(f"3 Reasons to sell {ticker} Stock")
    swot = get_chatgpt_response(f"SWOT analysis of {ticker} Stock")

    with st.beta_expander("OpenAI ChatGPT Analysis"):
        buy_reason, sell_reason, swot_analysis = st.tabs(["3 Reasons to Buy", "3 Reasons to Sell", "SWOT Analysis"])

        with buy_reason:
            st.subheader(f"3 Reasons to Buy {ticker} Stock")
            st.write(buy)

        with sell_reason:
            st.subheader(f"3 Reasons to Sell {ticker} Stock")
            st.write(sell)

        with swot_analysis:
            st.subheader(f"SWOT Analysis of {ticker} Stock")
            st.write(swot)
else:
    st.write("Please enter a valid ticker symbol in the sidebar.") 



with tech_indicator:
    import pandas_ta as ta
    st.subheader("Technical Analysis Dashboard: ")
    df = pd.DataFrame
    ind_list = df.ta.indicators(as_list=True)
    technical_indicator = st.selectbox("Tech Indicator", options=ind_list)
    method = technical_indicator
    indicator = pd.DataFrame(getattr(ta, method)(low=data["Low"], close=data["Close"], high=data["High"], open=data["Open"], volume=data["Volume"]))
    indicator["Close"] = data["Close"]
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)
'''

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

st.title("Stock Dashboard")

ticker = st.sidebar.text_input("Ticker")
strat_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

data = yf.download(ticker, start=strat_date, end=end_date)

if data.empty:
    st.error("No data available for the selected ticker.")
else:
    fig = px.line(data, x=data.index, y=data["Adj Close"], title=f"{ticker} Stock Price")
    st.plotly_chart(fig)

pricing_data, fundamental_data, news, tech_indicator, lstm_prediction = st.tabs(
    ["Pricing Data", "Fundamental Data", "Top 10 News", "Technical Analysis Dashboard", "LSTM Prediction"]
)

with pricing_data:
    if "Adj Close" in data.columns:
        data2 = data.copy()  
        data2["% Change"] = data2["Adj Close"] / data2["Adj Close"].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)  
        
        annual_return = data2["% Change"].mean() * 252 * 100
        st.write("Annual Return is ", annual_return, "%")
        stdev = np.std(data2["% Change"]) * np.sqrt(252)
        st.write("Standard Deviation is ", stdev * 100, "%")
        st.write("Risk Adjusted Return is ", annual_return / (stdev * 100))
    else:
        st.write("The 'Adj Close' column is missing in the data.")

with lstm_prediction:
    st.subheader("Stock Price Prediction using LSTM")
    if st.button("Predict Future Prices"):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Adj Close"].values.reshape(-1, 1))

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        x_train, y_train = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=50, batch_size=32)

        test_data = scaled_data[train_size - 60:]
        x_test, y_test = [], []

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            y_test.append(test_data[i, 0])

        x_test = np.array(x_test)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)  # Inverse scaling

        train = data[:train_size]
        valid = data[train_size:]
        valid['Predictions'] = predictions

        fig2 = px.line(valid, x=valid.index, y=["Adj Close", "Predictions"], title=f"{ticker} Stock Price Prediction")
        st.plotly_chart(fig2)

with fundamental_data:
    from alpha_vantage.fundamentaldata import FundamentalData
    key = "0M5NL6LVC4IF4ATX."
    fd = FundamentalData(key, output_format="Pandas")
    st.subheader("Balance Sheet")
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader("Income Statement")
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader("Cash Flow Statement")
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)

with news:
    from stocknews import StockNews
    st.header(f"News of {ticker}")
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f"News {i+1}")
        st.write(df_news["published"][i])
        st.write(df_news["title"][i])
        st.write(df_news["summary"][i])
        title_sentiment = df_news["sentiment_title"][i]
        st.write(f"Title Sentiment {title_sentiment}")
        news_sentiment = df_news["sentiment_summary"][i]
        st.write(f"News Sentiment {news_sentiment}")

import pandas_ta as ta

with tech_indicator:
    st.subheader("Technical Analysis Dashboard: ")
    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list=True)
    technical_indicator = st.selectbox("Tech Indicator", options=ind_list)
    method = technical_indicator
    indicator = pd.DataFrame(getattr(ta, method)(low=data["Low"], close=data["Close"], high=data["High"], open=data["Open"], volume=data["Volume"]))
    indicator["Close"] = data["Close"]
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)

