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
]

PREDICT_PARAM = 'Close'

NOT_NORMILIZE_PARAMS = [

]

PREDICTION_TIMES = 1
INTERVAL = "1d"
TEST_DAYS = 200
SEQ_LEN = 7
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

TRAIN_PARAMS = ['Open',
                'High',
                'Low',
                'Close',
                'Volume',
                'Price',
                'RSI_' + str(IND_HISTORY),
                'RSI_' + str(IND_HISTORY) + '_A_' + str(RSI_MAX),
                'RSI_' + str(IND_HISTORY) + '_B_' + str(RSI_MIN),
                'DifInPer']


'Close'
predicted = [[0.3430381 ],
    [0.34465432],
    [0.34301013],
    [0.35835037],
    [0.3785639 ],
    [0.3608193 ],
    [0.36318302],
    [0.36480173],
    [0.35713282],
    [0.36461207],
    [0.35658616],
    [0.378611  ],
    [0.36690146],
    [0.35874566],
    [0.36549202],
    [0.37214723],
    [0.38065884],
    [0.37876287],
    [0.3868341 ],
    [0.38595748],
    [0.35821763],
    [0.35755536],
    [0.3650699 ]]
'DifInPer'

predicted = [[-0.27321815],
    [-0.39876246],
    [-0.3977541 ],
    [ 0.20472887],
    [ 0.12922698],
    [ 0.5695647 ],
    [ 1.2677343 ],
    [ 1.2571496 ],
    [ 0.75532955],
    [ 0.99801666],
    [ 0.89433336],
    [-0.02415785],
    [ 0.02056827],
    [-0.48955345],
    [-0.44140142],
    [ 0.08009709],
    [-0.325729  ],
    [-0.6798023 ],
    [-0.00781771],
    [-0.45795888],
    [-0.6057623 ],
    [-0.5977494 ],
    [-0.48291498]]
"""
