from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

from env import TEST_DAYS, SEQ_LEN, OFFSET, OUT_PARAMS, NOT_NORMILIZE_PARAMS, PREDICT_PARAM
from dataManager import dropColumns


def normalization(data):
    notNormilizedColumns = data[NOT_NORMILIZE_PARAMS].copy()
    data = dropColumns(data, NOT_NORMILIZE_PARAMS)

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(x_scaled, index=data.index, columns=data.columns)
    data = pd.concat([data, notNormilizedColumns], axis=1)
    return data, scaler


def processData(X_data, Y_data, SEQ_LEN, OFFSET, OUT_PARAMS):

    np_in_data = np.array(X_data)
    np_out_data = np.array(Y_data).reshape(-1, 1)

    N = np_in_data.shape[0]
    k = N - (SEQ_LEN + OFFSET + OUT_PARAMS) + 1
    # Create Input and output Slice
    in_slice = np.array([range(i, i + SEQ_LEN) for i in range(k)])
    out_slice = np.array(
        [range(i + SEQ_LEN + OFFSET, i + SEQ_LEN + OFFSET + OUT_PARAMS) for i in range(k)])

    in_data = np_in_data[in_slice, :]
    out_data = np_out_data[out_slice, :]
    return in_data, out_data


def cutData(data, test_days, file_data_train, file_data_test):
    df = data.copy()
    train_data = df[:-test_days]
    df = data.copy()
    test_data = df.tail(test_days)
    file_data_train.write(str(train_data))
    file_data_test.write(str(test_data))
    return train_data, test_data


def prepareData(data, file_data_train, file_data_test, file_X_train, file_Y_train, file_X_test, file_Y_test, file_norm):
    data, scaler = normalization(data)
    file_norm.write(str(data))
    print("-----Data is Normalized------")

    train_data, test_data = cutData(
        data, TEST_DAYS, file_data_train, file_data_test)
    print("-----Created test and training data-----")

    X_train, Y_train = processData(
        train_data, train_data[PREDICT_PARAM], SEQ_LEN, OFFSET, OUT_PARAMS)
    X_test, Y_test = processData(
        test_data, test_data[PREDICT_PARAM], SEQ_LEN, OFFSET, OUT_PARAMS)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    file_X_train.write(str(X_train))
    file_Y_train.write(str(Y_train))
    file_X_test.write(str(X_test))
    file_Y_test.write(str(Y_test))
    print("------Train and test data processed-----")

    return data, scaler, train_data, test_data, X_train, Y_train, X_test, Y_test
