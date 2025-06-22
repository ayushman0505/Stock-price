import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.markdown(
    "Created and designed by <a href='https://www.linkedin.com/in/ayushman-raghuvanshi' target='_blank'>Ayushman Raghuvanshi</a>",
    unsafe_allow_html=True
)

def show_top_5_stocks():
    st.sidebar.header("Top 5 Performing Stocks (Last 30 Days)")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'JPM', 'V']
    nse_map = {
        'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'META': 'FB',
        'TSLA': 'TSLA', 'NVDA': 'NVDA', 'NFLX': 'NFLX', 'JPM': 'JPM', 'V': 'V'
    }
    bse_map = nse_map
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        returns = (data.iloc[-1] - data.iloc[0]) / data.iloc[0]
        top5 = returns.sort_values(ascending=False).head(5)
        for ticker, ret in top5.items():
            nse_url = f"https://www.nseindia.com/get-quotes/equity?symbol={nse_map[ticker]}"
            bse_url = f"https://www.bseindia.com/stock-share-price/stockreach_stockdetails.aspx?scripcode={bse_map[ticker]}"
            st.sidebar.markdown(
                f"<b>{ticker}</b>: {ret:.2%} &nbsp; "
                f"<a href='{nse_url}' target='_blank'>NSE</a> | "
                f"<a href='{bse_url}' target='_blank'>BSE</a>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.sidebar.write("Could not fetch top stocks.")

show_top_5_stocks()

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'], index=2)
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

@st.cache_data
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=300)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
        data = download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')
else:
    data = download_data(option, start_date, end_date)

scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    if data.empty or 'Close' not in data.columns or data['Close'].isnull().any():
        st.error('Data is not available or contains missing values.')
        return

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=20)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)   
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)

def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)}\nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

if __name__ =='__main__':
    main()

