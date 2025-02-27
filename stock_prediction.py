import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


st.title("Market Pulse:- Stay Ahead With Predictive Power")
stock = st.text_input("Enter the Stock ID", "GOOG")
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

data = yf.download(stock, start, end)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel('Ticker')

try:
    model = load_model("Latest_stock_price_model.keras")
except FileNotFoundError:
    st.error("Model file not found. Ensure 'Latest_stock_price_model.keras' is in the correct path.")
    st.stop()

st.subheader("Stock Data")
st.write(data)

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data.index, full_data['Close'], 'b', label="Original Close Price")
    plt.plot(full_data.index, values, 'orange', label="Moving Average")
    if extra_data and extra_dataset is not None:
        plt.plot(full_data.index, extra_dataset, 'g', label="Additional Moving Average")
    plt.legend()
    return fig

for ma in [250, 200, 100]:
    data[f'MA_for_{ma}_days'] = data['Close'].rolling(ma).mean()
    st.subheader(f'Original Close Price and MA for {ma} days')
    st.pyplot(plot_graph((15, 6), data[f'MA_for_{ma}_days'], data))

splitting_len = int(len(data) * 0.7)
x_test = pd.DataFrame(data['Close'][splitting_len:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame({
    'Original Test Data': inv_y_test.flatten(),
    'Predictions': inv_pre.flatten()
}, index=data.index[splitting_len + 100:])

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(data.index[:splitting_len + 100], data['Close'][:splitting_len + 100], label="Data - not used")
plt.plot(ploting_data['Original Test Data'], label="Original Test Data")
plt.plot(ploting_data['Predictions'], label="Predicted Test Data")
plt.legend()
st.pyplot(fig) 