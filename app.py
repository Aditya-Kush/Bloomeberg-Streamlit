import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier

st.set_page_config(page_title="Bloomberg Terminal Clone", layout="wide")

st.title("🏦 Bloomberg-Style Trading Terminal (India)")

nifty50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BHARTIARTL.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HCLTECH.NS",
    "WIPRO.NS", "TECHM.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS",
    "GRASIM.NS", "ADANIENT.NS", "ADANIPORTS.NS", "CIPLA.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BRITANNIA.NS", "BPCL.NS", "IOC.NS", "SHREECEM.NS",
    "UPL.NS"
]

@st.cache_data
def load(stock):
    df = yf.download(stock, period="2y", interval="1d", auto_adjust=True)

    if df.empty:
        return df

    df.columns = df.columns.get_level_values(0)

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    return df


def train(df):
    features = ["SMA_10","SMA_20","EMA_10","RSI","Lag_1","Lag_2","Volume"]

    X = df[features]
    y = df["Target"]

    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    model.fit(X, y)

    return model


def monte_carlo(price, prob, days=7, sims=200):
    paths = []

    for _ in range(sims):
        p = price
        path = [p]

        for _ in range(days):
            noise = np.random.normal(0, 0.01)
            drift = (prob - 0.5) * 0.025
            p = p * (1 + drift + noise)
            path.append(p)

        paths.append(path)

    return np.array(paths)


st.sidebar.header("📌 Stock Selector")
selected = st.sidebar.selectbox("Choose Stock", nifty50)

df = load(selected)

if not df.empty:

    model = train(df)
    latest = df.iloc[-1]

    X = np.array([[
        latest["SMA_10"],
        latest["SMA_20"],
        latest["EMA_10"],
        latest["RSI"],
        latest["Lag_1"],
        latest["Lag_2"],
        latest["Volume"]
    ]])

    prob = model.predict_proba(X)[0][1]
    price = float(latest["Close"])

    sim = monte_carlo(price, prob)

    median = np.median(sim[:, -1])
    low = np.percentile(sim[:, -1], 10)
    high = np.percentile(sim[:, -1], 90)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Price", round(price,2))
    col2.metric("UP Probability", round(prob,2))
    col3.metric("7D Median", round(median,2))
    col4.metric("Risk Range", f"{round(low,2)} - {round(high,2)}")

    if prob > 0.65:
        st.success("🔥 STRONG BULLISH SETUP (LONG)")
    elif prob < 0.4:
        st.error("⚠️ BEARISH / SHORT SETUP")
    else:
        st.warning("📊 SIDEWAYS / NO TRADE")

    st.subheader("📊 Market Scanner")

    results = []

    for stock in nifty50:
        try:
            df2 = load(stock)
            if df2.empty:
                continue

            model2 = train(df2)
            last = df2.iloc[-1]

            X2 = np.array([[
                last["SMA_10"],
                last["SMA_20"],
                last["EMA_10"],
                last["RSI"],
                last["Lag_1"],
                last["Lag_2"],
                last["Volume"]
            ]])

            p = model2.predict_proba(X2)[0][1]

            results.append([stock, last["Close"], round(p,2)])

        except:
            continue

    df_out = pd.DataFrame(results, columns=["Stock","Price","Prob"])
    df_out = df_out.sort_values("Prob", ascending=False)

    st.dataframe(df_out)

    st.subheader("🔥 Top Opportunities")
    st.dataframe(df_out.head(5))

    st.subheader("⚠️ Weak Stocks")
    st.dataframe(df_out.tail(5))

else:
    st.warning("No data available")
