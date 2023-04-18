import yfinance as yf
import pandas_ta as ta
import itertools

from env import TRAIN_PARAMS, IND_HISTORY, RSI_MAX, RSI_MIN


def dropColumns(data, columns):
    data = data.drop(columns, axis=1)
    return data


def getMainData(tickers, n, interval):
    data = yf.download(tickers=tickers, period=str(
        n)+"d", interval=interval, start="2016-01-01", end="2023-03-10")
    data.reset_index(inplace=True)
    return data


def getPrice(data):
    data['Price'] = (data['High'] + data['Low'])/2
    return data


def getRSI(data):
    data.ta.rsi(close=data["Close"], length=IND_HISTORY, append=True,
                signal_indicators=True, xa=RSI_MAX, xb=RSI_MIN)
    return data


def getMACD(data):
    data.ta.macd(close=data["Close"], fast=12, slow=26, signal=9, append=True)
    return data


def getATR(data):
    data["ATR"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)
    return data


def getBollingerBands(data):
    x = ta.bbands(data["Close"], length=20)
    data["BBL"] = x["BBL_20_2.0"]
    data["BBM"] = x["BBM_20_2.0"]
    data["BBU"] = x["BBU_20_2.0"]
    data["BBB"] = x["BBB_20_2.0"]
    data["BBP"] = x["BBP_20_2.0"]
    return data


def getMFI(data):
    data["MFI"] = ta.mfi(data["High"], data["Low"],
                         data["Close"], data["Volume"], length=14)
    return data


def getOBV(data):
    data["OBV"] = ta.obv(data["Close"], data["Volume"])
    return data


def getEMA(data):
    data["EMA"] = ta.ema(data["Close"], length=10)
    return data


def getADX(data):
    x = ta.adx(data["High"], data["Low"], data["Close"], length=14)
    data["ADX"] = x["ADX_14"]
    data["DMP"] = x["DMP_14"]
    data["DMN"] = x["DMN_14"]
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
    data = getMACD(data)

    data = getATR(data)
    data = getBollingerBands(data)

    data = getMFI(data)
    data = getOBV(data)

    data = getEMA(data)
    data = getADX(data)

    data = getDifInPer(data)
    data = data.dropna()

    listToDrop = list()
    for i in data.columns:
        if i not in TRAIN_PARAMS:
            listToDrop.append(i)

    data = dropColumns(data, listToDrop)

    file_data.write(str(data))
    return data
