import yfinance as yf
import pandas_ta as ta


def getRSI(data, length=14):
    data.ta.rsi(close=data["Close"], length=length, append=True,
                signal_indicators=True, xa=60, xb=40)
    data = data.iloc[length:]
    return data


def dropColumns(data, columns):
    data = data.drop(columns, axis=1)
    return data


def getData(file_data, params, tickers='BTC-USD', n=3000, interval="1d"):
    data = yf.download(tickers=tickers, period=str(n)+"d", interval=interval)
    data.reset_index(inplace=True)
    data['Price'] = (data['High'] + data['Low'])/2
    data = dropColumns(data, ['Date', 'Adj Close'])
    data = getRSI(data)
    file_data.write(str(data))
    return data
