import os
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from keras.layers import Bidirectional, Dropout, Activation, Dense
from keras.layers import CuDNNLSTM
import warnings
warnings.filterwarnings("ignore")

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

file_data = open("Data.txt", "w")
file_data_train = open("Data_train.txt", "w")
file_data_test = open("Data_test.txt", "w")
file_X_test = open("Data_X_test.txt", "w")
file_Y_test = open("Data_Y_test.txt", "w")
file_X_train = open("Data_X_train.txt", "w")
file_Y_train = open("Data_Y_train.txt", "w")
file_real_pred = open("Data_Real_Pred.txt", "w")

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8

TEST_DAYS = 90
SEQ_LEN = 30
OFFSET = 0
OUT_PARAMS = 1
scaler = StandardScaler()


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
                            input_shape=(WINDOW_SIZE, X_train.shape[2])))
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
    scaled_features = scaler.fit_transform(data.values)
    data = pd.DataFrame(
        scaled_features, index=data.index, columns=data.columns)
    return data


def processData(X_data, Y_data, SEQ_LEN, OFFSET, OUT_PARAMS):

    np_in_data = np.array(X_data)
    np_out_data = np.array(Y_data).reshape(-1, 1)

    N = np_in_data.shape[0]
    k = N - (SEQ_LEN + OFFSET + OUT_PARAMS)
    # Create Input and output Slice
    in_slice = np.array([range(i, i + SEQ_LEN) for i in range(k)])
    out_slice = np.array(
        [range(i + SEQ_LEN + OFFSET, i + SEQ_LEN + OFFSET + OUT_PARAMS) for i in range(k)])

    in_data = np_in_data[in_slice, :]
    out_data = np_out_data[out_slice, :]

    return in_data, out_data


def cutData(data, test_days):
    df = data.copy()
    train_data = df[:-test_days]
    df = data.copy()
    test_data = df.tail(test_days)
    file_data_train.write(str(train_data))
    file_data_test.write(str(test_data))
    return train_data, test_data


def getRealandPred(test_data, scaler, predicted, SEQ_LEN):
    temp = test_data.copy()
    temp = temp.iloc[SEQ_LEN:]
    temp = temp[:-1]

    x = scaler.inverse_transform(temp.values)
    temp = pd.DataFrame(
        x, index=temp.index, columns=temp.columns)
    realClosePrice = temp["Close"]

    temp = test_data.copy()
    temp = temp.iloc[SEQ_LEN:]
    temp = temp[:-1]

    predicted = predicted.flatten()

    temp = dropColumns(temp, ["Close"])
    temp = temp.assign(Close=predicted)

    x = scaler.inverse_transform(temp.values)
    temp = pd.DataFrame(
        x, index=temp.index, columns=temp.columns)

    df = {'Real': realClosePrice, 'Predicted': temp["Close"]}
    realPredPrice = pd.DataFrame(data=df)

    return realPredPrice


def createFinalGraf(df):
    plt.plot(df['Real'], label="Actual Price", color='green')
    plt.plot(df['Predicted'], label="Predicted Price", color='red')

    plt.title('Bitcoin price prediction')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')

    return plt


def calculateDifferencePercentage(realPredPrice):
    sum = 0
    count = 0
    min = 0
    max = 100
    dif = 0
    for index, row in realPredPrice.iterrows():
        a = row['Real']
        b = row['Predicted']

        if(a < b):
            dif = ((b-a)/a)*100
            if(dif > max):
                max = dif
            if(dif < min):
                min = dif
            sum += dif
        elif(a > b):
            dif = ((a-b)/a)*100
            if(dif > max):
                max = dif
            if(dif < min):
                min = dif
            sum += dif
            sum += dif
        else:
            sum += 0
        count += 1
    return round(sum/count, 2), min, max


def calculatePercentageofCorrectDirection(realPredPrice):
    df = realPredPrice.copy()
    for (columnName, columnData) in realPredPrice.iteritems():
        items = list(columnData.items())
        for i, values in items[:-1]:
            if(df.at[i, columnName] < df.at[i + 1, columnName]):
                df.loc[i, columnName] = 1
            else:
                df.loc[i, columnName] = 0
    df = df[:-1]

    correct = 0
    amount = df.shape[0]
    for i, row in df.iterrows():
        if(row['Real'] == row['Predicted']):
            correct += 1
    return round((correct*100)/amount, 2)


data = getData()
file_data.write(str(data))
print("-----Data Received-----")

data = dropColumns(data, ['Date', 'Adj Close'])
print("-----Columns Droped------")

data = normalization(data, scaler)
print("-----Data is Normalized------")

train_data, test_data = cutData(data, TEST_DAYS)
print("-----Created test and training data-----")

X_train, Y_train = processData(
    train_data, train_data['Close'], SEQ_LEN, OFFSET, OUT_PARAMS)
X_test, Y_test = processData(
    test_data, test_data['Close'], SEQ_LEN, OFFSET, OUT_PARAMS)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

file_X_train.write(str(X_train))
file_Y_train.write(str(Y_train))
file_X_test.write(str(X_test))
file_Y_test.write(str(Y_test))

print("------Train and test data processed-----")

model = createModel(SEQ_LEN, X_train)
print("------Model Created-----")

model = trainModel(model, X_train, Y_train)
print("------Model Trained-----")

model.evaluate(X_test, Y_test)
print("------Model Evaluated-----")

predicted = model.predict(X_test)
print("------Model Predicted-----")


"""
predicted = np.array([[1.0405878],
                      [1.0442121],
                      [1.052566],
                      [1.0658566],
                      [1.0922579],
                      [1.1319269],
                      [1.1713343],
                      [1.2268199],
                      [1.2844117],
                      [1.3454851],
                      [1.4145373],
                      [1.4751915],
                      [1.5140473],
                      [1.5308948],
                      [1.5336425],
                      [1.5351337],
                      [1.5367616],
                      [1.5363218],
                      [1.5339578],
                      [1.5388974],
                      [1.55428],
                      [1.5726322]])
"""

realPredPrice = getRealandPred(
    test_data, scaler, predicted, SEQ_LEN)

file_real_pred.write(str(realPredPrice))

print('average difference percentage = ' +
      str(calculateDifferencePercentage(realPredPrice)) + '%')
print('percantage of correct direction = ' +
      str(calculatePercentageofCorrectDirection(realPredPrice)) + '%')


plt = createFinalGraf(realPredPrice)

file_data.close()
file_data_train.close()
file_data_test.close()
file_X_test.close()
file_Y_test.close()
file_X_train.close()
file_Y_train.close()
file_real_pred.close()

plt.show()

print("-----Done-----")
