import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import yfinance as yf
import random

import matplotlib.pyplot as plt

from openpyxl import load_workbook

import warnings
warnings.filterwarnings("ignore")

''' ----- RECEIVE DATA -----'''

xlwriter = pd.ExcelWriter('historical prices.xlsx', engine='openpyxl')


tickerSymbol = 'BTC-USD'
tickerData = yf.Ticker(tickerSymbol)

data = tickerData.history(interval='1d', start='2014-09-17', end='2022-5-1')
data['price'] = (data['High'] + data['Low'])/2

print("1 ... DATA RECEIVED")

''' ----- PROCESS DATA -----'''

df = data[['price']]

df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
df['position'] = np.where(df['log_returns'] > 0, 1, 0)


def lag_column_creator(df, num_lags):
    for lag_num in range(1, num_lags + 1):
        df[f'lags_{lag_num}'] = df['log_returns'].shift(lag_num)
        df.dropna(inplace=True)
    return df

def getDataTV(symbol="BTCUSD", exchange="BINANCE", interval=Interval.in_daily, n=1000):
    tv = TvDatafeed(USERNAME, PASSWORD)
    data = tv.get_hist(symbol=symbol, exchange=exchange,
                       interval=interval, n_bars=n)
    data.reset_index(inplace=True)
    data = dropColumns(data, ['datetime', 'symbol'])
    data = data.rename(columns={'close': CLOSE})
    return data


def getData():
    csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    df['Price'] = (df['High'] + df['Low'])/2
    df = dropColumns(df, ['Date', 'Adj Close'])
    return df


df = lag_column_creator(df, 7)

df['momentum'] = df['log_returns'].rolling(5).mean().shift(1)
df['volatility'] = df['log_returns'].rolling(20).std().shift(1)
df['distance'] = (df['price'] - df['price'].rolling(50).mean()).shift(1)

df.dropna(inplace=True)



print("2 ... PROCESS DATA")

''' ----- CREATE MODEL -----'''

column_names = list(df.columns)[3:]

training_data = df[df.index < '2021-08-01'].copy()

mean, std = training_data.mean(), training_data.std()

training_data_ = (training_data - mean) / std

test_data = df[df.index >= '2021-08-01'].copy()
test_data_ = (test_data - mean) / std

random.seed(90)
np.random.seed(90)
tf.random.set_seed(90)

optimizer = Adam(learning_rate=0.0001)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(column_names), )))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(training_data_[column_names],
          training_data['position'], verbose=False, epochs=25)

print("3 ... MODEL CREATED")


''' ----- TEST MODEL -----'''

model.evaluate(training_data_[column_names], training_data['position'])

prediction = np.where(model.predict(test_data_[column_names]) > 0.5, 1, 0)

test_data['prediction'] = np.where(prediction > 0, 1, -1)

def normilizeData(df, scaler):
    close_price = df.Close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)
    return scaled_close


def to_sequences(data, seq_len):

    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)


def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

'''
test_data['log_returns_strategised'] = (
    test_data['prediction'] * test_data['log_returns'])

test_data[['log_returns', 'log_returns_strategised']].sum().apply(np.exp)
test_data[['log_returns', 'log_returns_strategised']
          ].cumsum().apply(np.exp).plot(figsize=(12, 7))
''' 
print("4 ... MODEL TESTED")

test_data['position'].replace(to_replace = 0, value = -1, inplace=True)

print(test_data)

plt.plot(range(len(test_data)), test_data['prediction'], label = "prediction")
plt.plot(range(len(test_data)), test_data['position'], label = "position")

plt.xlabel('Data')
plt.ylabel('Value')

plt.title('Predicted vs Momentum')

plt.legend()
plt.show()

test_data.to_excel(xlwriter, sheet_name=tickerSymbol, index=False)
xlwriter.save()
