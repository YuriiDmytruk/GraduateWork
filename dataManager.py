import yfinance as yf
import pandas_ta as ta
import itertools

from env import TRAIN_PARAMS, IND_HISTORY, RSI_MAX, RSI_MIN


def dropColumns(data, columns):
    data = data.drop(columns, axis=1)
    return data


def getMainData(tickers, n, interval):
    data = yf.download(tickers=tickers, period=str(n)+"d", interval=interval)
    data.reset_index(inplace=True)
    return data


def getPrice(data):
    data['Price'] = (data['High'] + data['Low'])/2
    return data


def getRSI(data):
    data.ta.rsi(close=data["Close"], length=IND_HISTORY, append=True,
                signal_indicators=True, xa=RSI_MAX, xb=RSI_MIN)
    return data


def getDifInPer(data):
    prev = 0
    res = list()
    for index, row in data.iterrows():
        cur = row['Close']
        if index == 0:
            res.append(None)
            prev = cur
        else:
            res.append(((cur - prev) * 100) / prev)
            prev = cur
    data["DifInPer"] = res
    return data


def getData(file_data, tickers='BTC-USD', n=3000, interval="1d"):
    data = getMainData(tickers, n, interval)
    data = getPrice(data)
    data = getRSI(data)
    data = getDifInPer(data)

    data = data.dropna()

    listToDrop = list()
    for i in data.columns:
        if i not in TRAIN_PARAMS:
            listToDrop.append(i)

    data = dropColumns(data, listToDrop)

    file_data.write(str(data))
    return data
