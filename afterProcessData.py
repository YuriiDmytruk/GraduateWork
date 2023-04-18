import pandas as pd
import numpy as np

from env import REAL, PREDICTED, NOT_NORMILIZE_PARAMS, PREDICT_PARAM, SEQ_LEN, CLOSE
from dataManager import dropColumns


def getTemp(test_data, x=True):
    temp = test_data.copy()
    temp = temp.iloc[SEQ_LEN:]
    if x:
        temp = dropColumns(temp, NOT_NORMILIZE_PARAMS)
    return temp


def inverse(scaler, temp):
    x = scaler.inverse_transform(temp.values)
    temp = pd.DataFrame(
        x, index=temp.index, columns=temp.columns)
    return temp


def getRealandPred(test_data, scaler, predicted):
    predicted = np.array(predicted)
    realPredPrice = afterProcess(test_data, scaler, predicted)

    if PREDICT_PARAM == 'DifInPer':
        realPredPrice = getNormalDifInPer(test_data, scaler, realPredPrice)
    return realPredPrice


def afterProcess(test_data, scaler, predicted):
    realPredPrice = pd.DataFrame()
    print("AFTER------------")
    if PREDICT_PARAM in NOT_NORMILIZE_PARAMS:
        print("IN________________")
        temp = getTemp(test_data, False)
        realPredPrice[REAL] = temp[PREDICT_PARAM].append(
            to_append=pd.Series([None]), ignore_index=True)
        """
        realPredPrice[PREDICTED] = pd.Series([None]).append(
            to_append=pd.Series(predicted.flatten()), ignore_index=True)
            """
    else:
        realPredPrice = denormilizeData(test_data, scaler, predicted)
    return realPredPrice


def denormilizeData(test_data, scaler, predicted):
    realPredPrice = pd.DataFrame()

    temp = getTemp(test_data)
    temp = inverse(scaler, temp)
    realPredPrice[REAL] = temp[PREDICT_PARAM].append(
        to_append=pd.Series([None]), ignore_index=True)

    temp = getTemp(test_data)
    predicted = predicted.flatten()
    temp[PREDICT_PARAM] = predicted[1:]
    temp = inverse(scaler, temp)
    realPredPrice[PREDICTED] = pd.Series([None]).append(
        to_append=temp[PREDICT_PARAM], ignore_index=True)

    diference = realPredPrice[PREDICTED][1] - realPredPrice[REAL][1]

    counter = 0
    for i in realPredPrice[PREDICTED]:
        if i is not None:
            realPredPrice[PREDICTED][counter] -= diference
        counter += 1

    return realPredPrice


def getNormalDifInPer(test_data, scaler, realPredPrice):
    if CLOSE in NOT_NORMILIZE_PARAMS:
        temp = getTemp(test_data, False)
    else:
        temp = getTemp(test_data)
        temp = inverse(scaler, temp)

    realPredPrice[REAL] = temp[CLOSE].append(
        to_append=pd.Series([None]), ignore_index=True)

    prevClose = 0
    res = list([realPredPrice[REAL].iloc[0]])
    for index, row in realPredPrice.iterrows():
        pred = row[PREDICTED]
        if pred is None:
            prevClose = res[0]
        else:
            predClose = ((prevClose / 100) * pred) + prevClose
            res.append(predClose)
            prevClose = predClose

    realPredPrice[PREDICTED] = pd.Series([None]).append(
        to_append=pd.Series(res), ignore_index=True)
    return realPredPrice


def calculateDifferencePercentage(realPredPrice):
    sum = 0
    count = 0
    min = 100
    max = 0
    dif = 0
    for index, row in realPredPrice.iterrows():
        a = row[REAL]
        b = row[PREDICTED]

        if a is not None and b is not None:
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
        for i, values in items[: -1]:
            if df.at[i, columnName] is not None and df.at[i + 1, columnName] is not None:
                if(df.at[i, columnName] < df.at[i + 1, columnName]):
                    df.loc[i, columnName] = 1
                else:
                    df.loc[i, columnName] = 0
    df = df[: -1]

    correct = 0
    amount = df.shape[0]
    for i, row in df.iterrows():
        if(row[REAL] == row[PREDICTED]):
            correct += 1
    return round((correct*100)/amount, 2)
