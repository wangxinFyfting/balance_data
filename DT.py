# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor

import pandas as pd
from collections import Counter

import os
def init(file_path):
    # Create a random dataset
    data = pd.read_csv(file_path)
    data['datetime'] = pd.to_datetime(data['datetime']).astype(int)
    X =data.drop("errorID", axis=1).values
    y = data["errorID"].values

    X_test = X[-20000:]
    y_test = y[-20000:]

    X = X[:-20000]
    y = y[:-20000]

    return X, y, X_test, y_test

def DT(X, y, X_test, y_test, max_depth=0):
    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=max_depth)
    # regr_2 = DecisionTreeRegressor(max_depth=15)
    regr_1.fit(X, y)
    # regr_2.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X_test)
    # y_2 = regr_2.predict(X_test)

    y_1_int = [round(i) for i in y_1]

    MSE = np.mean((y_test - y_1_int) ** 2)
    RMSE = np.sqrt(MSE)
    relative_deviation = y_test - y_1_int
    counter = Counter(relative_deviation)
    return counter, RMSE

def csv_init():
    # 初始化列名  
    columns = ['file_name'] + ['max_depth'] + [f'element_({i})' for i in range(-5, 6)] + ['RMSE']  
    
    # 初始化一个空的DataFrame  
    df = pd.DataFrame(columns=columns) 
    return df


def test_csv(df_head, file_path):
 
    X, y, X_test, y_test = init(file_path)
    # i = 100
    # with open("result.txt", 'w') as fd:
    #     # for i in range(1000):
    #     if i == 100:
    #         fd.write("====================\n")
    #         fd.write("max_depth:{}\n".format(str(i)))
    #         counter, rmse= DT(X, y, X_test, y_test, i)
    #         for element, count in counter.items():
    #             fd.write("      element:{}  counter:{}\n".format(element, count))
    #         fd.write("  RMSE:{}\n".format(rmse))

    for i in range(3,10000):
        df_temp = csv_init()
        df_temp.loc[0]=0
        df_temp['file_name'] = file_path
        df_temp["max_depth"] = i
        counter, rmse= DT(X, y, X_test, y_test, i)
        for element, count in counter.items():
            df_temp["element_({})".format(str(element))] = count
        df_temp["RMSE"] = rmse
        df_temp.fillna(0, inplace=True)
        df_head.loc[-1] = df_temp.values[0]
        df_head = df_head.reset_index(drop=True)
        df_head.to_csv("test_result.csv", index=False)
        print(df_head)
# print(len(X_test))
# print(len(y))
# # Plot the results
# plt.figure()
# plt.scatter(np.arange(len(X_test)), y_test, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(np.arange(len(X_test)), y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(np.arange(len(X_test)), y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()

if "__main__" == __name__:

    file_list = ["merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_0_balance_object.csv",
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_2_balance_SMOTE.csv",
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_0.csv",
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_2.csv",
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_1_balance_SMOTE.csv",   
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_4_balance_SMOTE.csv",
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_1.csv",
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_4.csv]"]
    
    # 文件名  
    filename = 'test_result.csv'  
    df_head = pd.DataFrame()
    # 检查文件是否存在  
    if os.path.exists(filename):  
        # 文件存在，读取到DataFrame  
        df_head = pd.read_csv(filename)  
    else:
        df_head = csv_init()

    for file in file_list:
        test_csv(df_head, "./Orignal/" + file)