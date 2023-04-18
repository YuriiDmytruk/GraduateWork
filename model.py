from tensorflow import keras
from keras.layers import Bidirectional, Dropout, Activation, Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns
from pylab import rcParams

from env import PREDICTION_TIMES, REAL, PREDICTED, SEQ_LEN
from afterProcessData import getRealandPred, calculateDifferencePercentage, calculatePercentageofCorrectDirection

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8


def createFinalGraf(df):
    plt.plot(df[REAL], label="Actual Price", color='green')
    plt.plot(df[PREDICTED], label="Predicted Price", color='red')

    plt.title('Bitcoin price prediction')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')

    return plt


def createModel(X_train, DROPOUT=0.2):
    WINDOW_SIZE = SEQ_LEN

    model = keras.Sequential()

    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                            input_shape=(WINDOW_SIZE, X_train.shape[2])))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

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
        epochs=30,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.1
    )

    return model


def prepareModel(key, X_train, Y_train, X_test, Y_test):
    if(key == "LOAD"):
        model = keras.models.load_model("Core")
        print("------Model Loaded-----")
    else:
        model = createModel(X_train)
        print("------Model Created-----")

        model = trainModel(model, X_train, Y_train)
        print("------Model Trained-----")

        # model.save("Core")
    return model


def predict(model, X_test, file_pred, file_real_pred, test_data, scaler):
    predicted = model.predict(X_test)

    print("------Model Predicted-----")

    file_pred.write(str(predicted))

    realPredPrice = getRealandPred(
        test_data, scaler, predicted)

    file_real_pred.write(str(realPredPrice))

    #dif, min, max = calculateDifferencePercentage(realPredPrice)
    dif = 0
    min = 0
    max = 0
    percents = calculatePercentageofCorrectDirection(realPredPrice)

    plt = createFinalGraf(realPredPrice)

    return plt, dif, min, max, percents


def predictModel(key, X_train, Y_train, X_test, Y_test, file_pred, file_real_pred, test_data, scaler):

    df = pd.DataFrame([], columns=['diference', 'min', 'max', 'percents'])
    for i in range(1, PREDICTION_TIMES + 1):
        model = prepareModel(key, X_train, Y_train, X_test, Y_test)
        plt, dif, min, max, percents = predict(
            model, X_test, file_pred, file_real_pred, test_data, scaler)

        df.loc[len(df.index)] = [dif, min, max, percents]
    return plt, df
