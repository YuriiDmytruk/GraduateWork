import os
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from keras.layers import CuDNNLSTM
from keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")

# np.set_printoptions(threshold=sys.maxsize)

file_data = open("Data.txt", "w")
file_data_train = open("Data_train.txt", "w")
file_data_test = open("Data_test.txt", "w")
file_np_data = open("Data_np.txt", "w")
file_in_data = open("Data_in.txt", "w")
file_out_data = open("Data_out.txt", "w")
file_prediction = open("Data_predictio.txt", "w")
file_real = open("Data_real.txt", "w")

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

scaler = MinMaxScaler()

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

SEQ_LEN = 7


def getData():
    csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    df['Price'] = (df['High'] + df['Low'])/2
    return df


def createGraf(df):
    ax = df.plot(x='Date', y='Close')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    plt.show()


def createModel(SEQ_LEN, X_train, DROPOUT=0.2):
    WINDOW_SIZE = SEQ_LEN
    print((X_train.shape[-1]))

    model = keras.Sequential()

    model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=True),
                            input_shape=(WINDOW_SIZE, SEQ_LEN-1)))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(CuDNNLSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=False)))

    model.add(Dense(units=1))

    model.add(Activation('linear'))
    return model


def trainModel(model, X_train, Y_train):
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    BATCH_SIZE = 64

    history = model.fit(
        X_train,
        Y_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.1
    )

    return model


def dropColumns(data, columns):
    data = data.drop(columns, axis=1)
    return data


def normalization(data, scaler):
    scaled_features = StandardScaler().fit_transform(data.values)
    data = pd.DataFrame(
        scaled_features, index=data.index, columns=data.columns)
    return data


def processData(X_data=None, Y_data=None, days=30, offset=0, out_params=1):

    np_in_data = np.array(X_data)
    np_out_data = np.array(Y_data).reshape(-1, 1)
    file_np_data.write(np.array_str(np_in_data, precision=1,
                                    suppress_small=True) + str(np_in_data.shape))
    N = np_in_data.shape[0]
    k = N - (days + offset + out_params)
    # Create Input and output Slice
    in_slice = np.array([range(i, i + days) for i in range(k)])
    out_slice = np.array(
        [range(i + days + offset, i + days + offset + out_params) for i in range(k)])

    in_data = np_in_data[in_slice, :]
    file_in_data.write(np.array_str(in_data) + str(in_data.shape))

    out_data = np_out_data[out_slice, :]
    file_out_data.write(np.array_str(out_data) + str(out_data.shape))

    return in_data, out_data


def cutData(data, test_days):
    df = data.copy()
    train_data = df[:-test_days]
    df = data.copy()
    test_data = df.tail(test_days)
    file_data_train.write(str(train_data))
    file_data_test.write(str(test_data))
    return train_data, test_data


data = getData()
file_data.write(str(data))
print("-----Data Received-----")

data = dropColumns(data, ['Date', 'Adj Close'])
print("-----Columns Droped------")

data = normalization(data, scaler)
print("-----Data is Normalized------")

train_data, test_data = cutData(data, 30)
print("-----Created test and training data-----")

X_train, Y_train = processData(train_data, train_data['Close'], SEQ_LEN)
X_test, Y_test = processData(test_data, test_data['Close'], SEQ_LEN)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print("------Train and test data processed-----")

model = createModel(SEQ_LEN, X_train)
print("------Model Created-----")


model = trainModel(model, X_train, Y_train)

model.evaluate(X_test, Y_test)

predicted = model.predict(X_test)


file_prediction.write(str(predicted))
file_real.write(str(Y_test))
"""
predicted = np.array([[1.0405878]
                      [1.0442121]
                      [1.052566]
                      [1.0658566]
                      [1.0922579]
                      [1.1319269]
                      [1.1713343]
                      [1.2268199]
                      [1.2844117]
                      [1.3454851]
                      [1.4145373]
                      [1.4751915]
                      [1.5140473]
                      [1.5308948]
                      [1.5336425]
                      [1.5351337]
                      [1.5367616]
                      [1.5363218]
                      [1.5339578]
                      [1.5388974]
                      [1.55428]
                      [1.5726322]])
y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')

plt.show()
"""

print("done")
file_data.close()
file_in_data.close()
file_np_data.close()
file_data.close()
file_data_test.close()
file_data_train.close()
file_prediction.close()
file_real.close()
