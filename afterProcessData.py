import pandas as pd

from env import CLOSE, REAL, PREDICTED


def getRealandPred(test_data, scaler, predicted, SEQ_LEN):
    temp = test_data.copy()
    temp = temp.iloc[SEQ_LEN:]
    temp = temp[:-1]
    x = scaler.inverse_transform(temp.values)
    temp = pd.DataFrame(
        x, index=temp.index, columns=temp.columns)
    realClosePrice = temp[CLOSE]

    temp = test_data.copy()
    temp = temp.iloc[SEQ_LEN:]
    temp = temp[:-1]
    predicted = predicted.flatten()
    temp = temp.assign(Close=predicted)
    x = scaler.inverse_transform(temp.values)
    temp = pd.DataFrame(
        x, index=temp.index, columns=temp.columns)
    df = {REAL: realClosePrice, PREDICTED: temp[CLOSE]}
    realPredPrice = pd.DataFrame(data=df)

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
        if(row[REAL] == row[PREDICTED]):
            correct += 1
    return round((correct*100)/amount, 2)
