import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
import yfinance as yf
from openpyxl import load_workbook
import pandas as pd
import warnings
import random

warnings.filterwarnings("ignore")

tikerSymbol = 'BTC-USD'

tickerData = yf.Ticker(tikerSymbol)
data = tickerData.history(interval='1d', start='2010-1-1', end='2022-5-1')

data['price'] = (data['High'] + data['Low'])/2

df = data[['price']]

df['log_returns'] = np.log(df['price'] / df['price'].shift(1))

df['position'] = np.where(df['log_returns'] > 0, 1, 0)

def lag_column_creator(df, num_lag):
    for lag_num in range(1, num_lag + 1):
        df[f'lags_{lag_num}'] = df['log_returns'].shift(lag_num)
        df.dropna(inplace=True)
    return df

df = lag_column_creator(df, 7)

df['momentum'] = df['log_returns'].rolling(5).mean().shift(1)
df['volatility'] = df['log_returns'].rolling(20).std().shift(1)
df['distance'] = (df['price'] - df['price'].rolling(50).mean()).shift(1)

df.dropna(inplace=True)

column_names = list(df.columns)[3:]

training_data = df[df.index < '2021-08-01'].copy()

mean, std = training_data.mean(), training_data.std()

training_data_ = (training_data - mean) / std

test_data = df[df.index >= '2021-08-01'].copy()
test_data = (test_data - mean) / std

random.seed(90)
np.random.seed(90)
tf.random.set_seed(90)

optimizer = Adam(learning_rate=0.0001)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(column_names), )))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizer, loss='binary_crosentropy', metrics=['accuracy'])

model.fit(training_data_[column_names], training_data['position'], verbose=False, epochs=25)

model.evaluate(training_data_[column_names], training_data['position'])


# print(column_names)

#xlwriter = pd.ExcelWriter('historical prices.xlsx', engine='openpyxl')
#df.to_excel(xlwriter, sheet_name=tikerSymbol, index=False)

#xlwriter.save()
