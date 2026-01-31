import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ===================== UI =====================
st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.markdown(
    "Created and designed by <a href='https://www.linkedin.com/in/ayushman-raghuvanshi' target='_blank'>Ayushman Raghuvanshi</a>",
    unsafe_allow_html=True
)

# ===================== STOCK SELECTION =====================
exchange = st.sidebar.selectbox('Select Exchange', ['NSE', 'BSE'])

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

option = st.sidebar.selectbox('Enter a Stock Symbol', stock_options[exchange])
today = datetime.date.today()

duration = st.sidebar.number_input('Enter the duration', value=300)
start_date = st.sidebar.date_input('Start Date', today - datetime.timedelta(days=duration))
end_date = st.sidebar.date_input('End Date', today)

data = yf.download(option, start=start_date, end=end_date, progress=False)

# ===================== TECHNICAL INDICATORS =====================
def tech_indicators():
    st.header('Technical Indicators')

    if data.empty or data['Close'].isnull().any():
        st.error("Data unavailable")
        return

    choice = st.radio('Choose Indicator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    if choice == 'Close':
        st.line_chart(data['Close'])

    elif choice == 'BB':
        bb = BollingerBands(data['Close'])
        df = pd.DataFrame({
            'Close': data['Close'],
            'High': bb.bollinger_hband(),
            'Low': bb.bollinger_lband()
        })
        st.line_chart(df)

    elif choice == 'MACD':
        st.line_chart(MACD(data['Close']).macd())

    elif choice == 'RSI':
        st.line_chart(RSIIndicator(data['Close']).rsi())

    elif choice == 'SMA':
        st.line_chart(SMAIndicator(data['Close']).sma_indicator())

    elif choice == 'EMA':
        st.line_chart(EMAIndicator(data['Close']).ema_indicator())

# ===================== DATAFRAME =====================
def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

# ===================== MODEL ENGINE =====================
def model_engine(model, num, return_score=False, show_forecast=False):
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num)
    df.dropna(inplace=True)

    if len(df) <= num:
        return (None, None) if return_score else None

    X = df[['Close']].values
    y = df['preds'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_forecast = X[-num:]
    X = X[:-num]
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    if show_forecast:
        forecast = model.predict(X_forecast)
        for i, val in enumerate(forecast, 1):
            st.text(f"Day {i}: {val:.2f}")

    if return_score:
        return r2, mae

# ===================== PREDICTION =====================
def predict():
    st.header('Stock Price Prediction')

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(),
        'ExtraTrees': ExtraTreesRegressor(),
        'KNN': KNeighborsRegressor(),
        'XGBoost': XGBRegressor()
    }

    num = int(st.number_input('How many days forecast?', value=20))

    if st.button('Predict'):
        scores = {}

        for name, model in models.items():
            scores[name] = model_engine(model, num, return_score=True)

        valid_scores = {k: v for k, v in scores.items() if v[0] is not None}

        if not valid_scores:
            st.error("Not enough data to evaluate models.")
            return

        best_model = max(valid_scores, key=lambda k: valid_scores[k][0])
        worst_model = min(valid_scores, key=lambda k: valid_scores[k][0])

        for name, (r2, mae) in scores.items():
            if r2 is None:
                st.write(f"**{name}**: Not enough data")
                continue

            tag = ""
            if name == best_model:
                tag = " (Recommended)"
            elif name == worst_model:
                tag = " (Least Recommended)"

            st.write(f"**{name}** → R²: {r2:.4f} | MAE: {mae:.4f}{tag}")

        st.subheader(f"Forecast using {best_model}")
        model_engine(models[best_model], num, show_forecast=True)

# ===================== MAIN =====================
def main():
    choice = st.sidebar.selectbox('Make a choice', ['Predict', 'Recent Data', 'Visualize'])

    if choice == 'Predict':
        predict()
    elif choice == 'Recent Data':
        dataframe()
    else:
        tech_indicators()

if __name__ == "__main__":
    main()
