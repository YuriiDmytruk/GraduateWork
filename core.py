import sys
import numpy as np
import warnings

from env import INTERVAL
from dataManager import getData
from dataProcess import prepareData
from model import predictModel

warnings.filterwarnings("ignore")

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


file_data = open("./Data/Data.txt", "w")
file_data_train = open("./Data/Data_train.txt", "w")
file_data_test = open("./Data/Data_test.txt", "w")
file_X_test = open("./Data/Data_X_test.txt", "w")
file_Y_test = open("./Data/Data_Y_test.txt", "w")
file_X_train = open("./Data/Data_X_train.txt", "w")
file_Y_train = open("./Data/Data_Y_train.txt", "w")
file_real_pred = open("./Data/Data_Real_Pred.txt", "w")
file_pred = open("./Data/Data_Pred.txt", "w")
file_norm = open("./Data/Data_Normilized.txt", "w")


data = getData(file_data, interval=INTERVAL)
print("-----Data Received-----")

data, scaler, train_data, test_data, X_train, Y_train, X_test, Y_test = prepareData(
    data, file_data_train, file_data_test, file_X_train, file_Y_train, file_X_test, file_Y_test, file_norm)
print('-----Data Prepared-----')

plt, outData = predictModel("CREATE", X_train, Y_train, X_test,
                            Y_test, file_pred, file_real_pred, test_data, scaler)

file_data.close()
file_data_train.close()
file_data_test.close()
file_X_test.close()
file_Y_test.close()
file_X_train.close()
file_Y_train.close()
file_real_pred.close()
file_pred.close()

print(str(outData))
plt.show()

print("-----Done-----")
