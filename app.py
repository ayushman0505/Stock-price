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

# Exchange selection
exchange = st.sidebar.selectbox('Select Exchange', ['NSE', 'BSE'])
default_symbol = 'TATAMOTORS.NS' if exchange == 'NSE' else 'TATAMOTORS.BO'

# Indian stock options for user
stock_options = {
    'NSE': [
        'TATAMOTORS.NS', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'ITC.NS', 'LT.NS'
    ],
    'BSE': [
        'TATAMOTORS.BO', 'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'INFY.BO',
        'ICICIBANK.BO', 'SBIN.BO', 'BHARTIARTL.BO', 'HINDUNILVR.BO', 'ITC.BO', 'LT.BO'
    ]
}

option = st.sidebar.selectbox('Enter a Stock Symbol', stock_options[exchange], index=0)
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=300)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
        data = yf.download(option, start=start_date, end=end_date, progress=False)
    else:
        st.sidebar.error('Error: End date must fall after start date')
        data = yf.download(option, start=start_date, end=end_date, progress=False)
else:
    data = yf.download(option, start=start_date, end=end_date, progress=False)

scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    if data.empty or 'Close' not in data.columns or data['Close'].isnull().any():
        st.error('Data is not available or contains missing values.')
        return

    option_tech = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])
    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data.copy()
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option_tech == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option_tech == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option_tech == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option_tech == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option_tech == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict():
    st.header('Stock Price Prediction')
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'XGBoostRegressor': XGBRegressor()
    }
    num = st.number_input('How many days forecast?', value=20)
    num = int(num)
    scores = {}
    if st.button('Predict'):
        for name, model in models.items():
            score = model_engine(model, num, return_score=True)
            scores[name] = score
        # Find best and worst
        best_model = max(scores, key=lambda k: scores[k][0] if scores[k][0] is not None else float('-inf'))
        worst_model = min(scores, key=lambda k: scores[k][0] if scores[k][0] is not None else float('inf'))
        for name, score in scores.items():
    r2, mae = score

    if r2 is None or mae is None:
        st.write(f"**{name}**: Not enough data to evaluate")
        continue

    rec = ""
    if name == best_model:
        rec = " (Recommended)"
    elif name == worst_model:
        rec = " (Least Recommended)"

    st.write(f"**{name}**: r2_score: {r2:.4f} | MAE: {mae:.4f}{rec}")
        # Show forecast for best model
        st.subheader(f"Forecast for {best_model}")
        model_engine(models[best_model], num, show_forecast=True)

def model_engine(model, num, return_score=False, show_forecast=False):
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num)

    #  DROP NaNs created by shifting
    df.dropna(inplace=True)

    if len(df) <= num:
        if return_score:
            return (None, None)
        st.error("Not enough data for the selected forecast period.")
        return

    X = df[['Close']].values
    y = df['preds'].values

    # Local scaler (NOT global)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_forecast = X[-num:]
    X = X[:-num]
    y = y[:-num]

    if len(X) == 0 or len(y) == 0:
        if return_score:
            return (None, None)
        return

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    if show_forecast:
        forecast_pred = model.predict(X_forecast)
        for day, val in enumerate(forecast_pred, 1):
            st.text(f'Day {day}: {val:.2f}')

    if return_score:
        return (r2, mae)

def show_top_5_stocks():
    st.sidebar.markdown("---")
    st.sidebar.header("Top 5 Performing Stocks (Last 30 Days)")
    tickers = stock_options[exchange]
    nse_map = {
        'TATAMOTORS.NS': 'TATAMOTORS', 'RELIANCE.NS': 'RELIANCE', 'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFCBANK', 'INFY.NS': 'INFY',
        'ICICIBANK.NS': 'ICICIBANK', 'SBIN.NS': 'SBIN', 'BHARTIARTL.NS': 'BHARTIARTL', 'HINDUNILVR.NS': 'HINDUNILVR', 'ITC.NS': 'ITC', 'LT.NS': 'LT'
    }
    bse_map = {
        'TATAMOTORS.BO': '500570', 'RELIANCE.BO': '500325', 'TCS.BO': '532540', 'HDFCBANK.BO': '500180', 'INFY.BO': '500209',
        'ICICIBANK.BO': '532174', 'SBIN.BO': '500112', 'BHARTIARTL.BO': '532454', 'HINDUNILVR.BO': '500696', 'ITC.BO': '500875', 'LT.BO': '500510'
    }
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)
    try:
        data5 = yf.download(tickers, start=start, end=end)['Adj Close']
        returns = (data5.iloc[-1] - data5.iloc[0]) / data5.iloc[0]
        top5 = returns.sort_values(ascending=False).head(5)
        for ticker, ret in top5.items():
            if exchange == 'NSE':
                nse_url = f"https://www.nseindia.com/get-quotes/equity?symbol={nse_map[ticker]}"
                bse_url = "#"
            else:
                nse_url = "#"
                bse_url = f"https://www.bseindia.com/stock-share-price/stockreach_stockdetails.aspx?scripcode={bse_map[ticker]}"
            st.sidebar.markdown(
                f"<b>{ticker}</b>: {ret:.2%} &nbsp; "
                f"<a href='{nse_url}' target='_blank'>NSE</a> | "
                f"<a href='{bse_url}' target='_blank'>BSE</a>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.sidebar.write("Could not fetch top stocks.")

def main():
    option_main = st.sidebar.selectbox('Make a choice', ['Recent Data', 'Predict', 'Visualize'], index=1)
    if option_main == 'Visualize':
        tech_indicators()
    elif option_main == 'Recent Data':
        dataframe()
    else:
        predict()

show_top_5_stocks()

if __name__ == '__main__':
    main()
