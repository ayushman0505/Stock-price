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

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Ayushman Raghuvanshi](www.linkedin.com/in/ayushman-raghuvanshi)")

def main():
    exchange = st.sidebar.selectbox('Select Exchange', ['NSE', 'BSE'], key='exchange_selectbox')
    stock_symbol = st.sidebar.text_input('Enter a Stock Symbol', value='TATAMOTORS', key='stock_symbol_input')
    stock_symbol = stock_symbol.upper()
    if exchange == 'NSE':
        stock_symbol += '.NS'
    elif exchange == 'BSE':
        stock_symbol += '.BO'
    savings = st.sidebar.number_input('Enter your monthly savings', value=1000, key='savings_input')
    st.sidebar.header('Top 5 Performing Stocks')
    top_stocks = get_top_stocks(exchange)
    for index, row in top_stocks.iterrows():
        st.sidebar.markdown(f"[{row['Stock']}]({row['Link']}) - {row['Close Price']}")
    predict(stock_symbol, savings, exchange)

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def get_top_stocks(exchange):
    if exchange == 'NSE':
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
    else:
        symbols = ['RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'INFY.BO', 'HINDUNILVR.BO']
    data = yf.download(symbols, period='1d', progress=False)
    close_prices = data['Close'].iloc[-1]
    top_stocks = close_prices.sort_values(ascending=False).head(5)
    top_stocks_df = pd.DataFrame(top_stocks).reset_index()
    top_stocks_df.columns = ['Stock', 'Close Price']
    top_stocks_df['Link'] = top_stocks_df['Stock'].apply(lambda x: f"https://www.nseindia.com/get-quotes/equity?symbol={x.split('.')[0]}" if exchange == 'NSE' else f"https://www.bseindia.com/stock-share-price/{x.split('.')[0]}")
    return top_stocks_df

today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000, key='duration_input')
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before, key='start_date_input')
end_date = st.sidebar.date_input('End date', today, key='end_date_input')
if st.sidebar.button('Send', key='send_button'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        stock_symbol = st.sidebar.text_input('Enter a Stock Symbol', value='TATAMOTORS', key='stock_symbol_input_send')
        stock_symbol = stock_symbol.upper()
        exchange = st.sidebar.selectbox('Select Exchange', ['NSE', 'BSE'], key='exchange_selectbox_send')
        if exchange == 'NSE':
            stock_symbol += '.NS'
        elif exchange == 'BSE':
            stock_symbol += '.BO'
        download_data(stock_symbol, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

def tech_indicators(stock_symbol):
    data = download_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.error("No data available for the given stock symbol and date range.")
        return
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'], key='tech_indicators_radio')

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

def dataframe(stock_symbol):
    data = download_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.error("No data available for the given stock symbol and date range.")
        return
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict(stock_symbol, savings, exchange):
    data = download_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.error("No data available for the given stock symbol and date range.")
        return
    scaler = StandardScaler()
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'], key='model_radio')
    num = st.number_input('How many days forecast?', value=5, key='num_days_input')
    num = int(num)
    if st.button('Predict', key='predict_button'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num, data, scaler, savings, stock_symbol, exchange)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num, data, scaler, savings, stock_symbol, exchange)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num, data, scaler, savings, stock_symbol, exchange)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num, data, scaler, savings, stock_symbol, exchange)
        else:
            engine = XGBRegressor()
            model_engine(engine, num, data, scaler, savings, stock_symbol, exchange)

def model_engine(model, num, data, scaler, savings, stock_symbol, exchange):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # dropping rows with NaN values
    df.dropna(inplace=True)
    if df.empty:
        st.error("Not enough data to make predictions.")
        return
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
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    forecast_dates = pd.date_range(start=today, periods=num).tolist()
    forecast_prices = []
    for i in forecast_pred:
        forecast_prices.append(i)

    # Plotting the predictions
    st.header('Predicted Stock Prices')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': forecast_prices})
    st.line_chart(forecast_df.set_index('Date'))

    # Displaying the predicted prices for each day
    st.header('Predicted Prices for Each Day')
    for date, price in zip(forecast_dates, forecast_prices):
        st.text(f'{date.date()}: {price}')

    # Calculating the number of stocks you can buy with your savings
    st.header('Number of Stocks You Can Buy with Your Savings')
    num_stocks = [savings / price for price in forecast_prices]
    for date, stocks in zip(forecast_dates, num_stocks):
        st.text(f'{date.date()}: {stocks} stocks')

    # Adding the "Invest Now" button
    st.header('Invest Now')
    if exchange == 'NSE':
        stock_page_url = f"https://www.nseindia.com/get-quotes/equity?symbol={stock_symbol.split('.')[0]}"
    else:
        stock_page_url = f"https://www.bseindia.com/stock-share-price/{stock_symbol.split('.')[0]}"
    st.markdown(f"[Click here to invest in {stock_symbol.split('.')[0]}]({stock_page_url})", unsafe_allow_html=True)

if __name__ == '__main__':
    main()



