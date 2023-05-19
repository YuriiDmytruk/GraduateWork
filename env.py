import numpy as np

USERNAME = "YuriiDmytruk"
PASSWORD = "letmeTrade1t"

CLOSE = "Close"
REAL = "Real"
PREDICTED = "Predicted"

IND_HISTORY = 14
RSI_MAX = 70
RSI_MIN = 30

TRAIN_PARAMS = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'Price',

    "BBL", "BBM", "BBU", "BBB", "BBP",

    'RSI_' + str(IND_HISTORY),
    'RSI_' + str(IND_HISTORY) + '_A_' + str(RSI_MAX),
    'RSI_' + str(IND_HISTORY) + '_B_' + str(RSI_MIN),
]

PREDICT_PARAM = 'Close'

NOT_NORMILIZE_PARAMS = [
    'RSI_' + str(IND_HISTORY) + '_A_' + str(RSI_MAX),
    'RSI_' + str(IND_HISTORY) + '_B_' + str(RSI_MIN),
]

PREDICTION_TIMES = 1
INTERVAL = "1d"
TEST_DAYS = 100
SEQ_LEN = 3
OFFSET = 0
OUT_PARAMS = 1


"""
    'ADX', 'DMP', 'DMN',

    "EMA"

    "OBV",

    "MFI",

    "ATR",

    "BBL", "BBM", "BBU", "BBB", "BBP",

    'RSI_' + str(IND_HISTORY),
    'RSI_' + str(IND_HISTORY) + '_A_' + str(RSI_MAX),
    'RSI_' + str(IND_HISTORY) + '_B_' + str(RSI_MIN),

    'MACD_12_26_9',
    'MACDh_12_26_9',
    'MACDs_12_26_9',

TEST_DAYS = 30
SEQ_LEN = 7

"""
