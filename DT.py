# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor

import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
import os

mod_deep_learn = {
    "GB" :None,
    "LR" :None,
    "DT": None,
    "KNN" :None,
    "RF" :None,
    "SVM" : None,
    "NN" : None
}
def int_mode(seednum_set):
    # 建立模型 
    mod_deep_learn["GB"] = GradientBoostingRegressor(random_state=seednum_set) 
    # https://blog.csdn.net/VariableX/article/details/107200334

    mod_deep_learn["LR"] = LogisticRegression(random_state=seednum_set)
    # https://blog.csdn.net/weixin_52589734/article/details/115005455

    mod_deep_learn["DT"] = DecisionTreeRegressor(max_depth=50, random_state=seednum_set)# max_depth=5, min_samples_leaf=4,min_samples_split=2
    # https://blog.csdn.net/qq_16000815/article/details/80954039

    mod_deep_learn["KNN"] = neighbors.KNeighborsRegressor()# algorithm='kd_tree', n_neighbors=5
    # https://blog.csdn.net/weixin_42182448/article/details/88639391

    mod_deep_learn["RF"] = RandomForestRegressor(random_state=seednum_set) #n_estimators=200, random_state= 8
    # https://blog.csdn.net/c1z2w3456789/article/details/104880683
    # https://www.codeleading.com/article/73814095186/  （默认参数设置）

    mod_deep_learn["SVM"] = svm.SVR() #C=1,kernel='rbf',gamma=0.2,decision_function_shape='ovr'
    # https://blog.csdn.net/github_39261590/article/details/75009069

    mod_deep_learn["NN"] = MLPRegressor(hidden_layer_sizes=(5,10,10,1),random_state = seednum_set) #用和D network相同结构的网络，设置相同的random_state)
    # https://blog.csdn.net/weixin_44022515/article/details/103286222

def init(file_path):
    # Create a random dataset
    data = pd.read_csv(file_path)
    data['datetime'] = pd.to_datetime(data['datetime']).astype(int)

    X =data.drop("errorID", axis=1).values
    y = data["errorID"].values.astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = X[-20000:]
    y_test = y[-20000:]

    X = X[:-20000]
    y = y[:-20000]

    return X, y, X_test, y_test

def DT(X, y, X_test, y_test, mode):
    # Fit regression model
    regr_1 = mod_deep_learn[mode]
    regr_1.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X_test)

    y_1_int = [round(i) for i in y_1]

    MSE = np.mean((y_test - y_1_int) ** 2)
    RMSE = np.sqrt(MSE)
    relative_deviation = y_test - y_1_int
    counter = Counter(relative_deviation)
    return counter, RMSE

def csv_init():
    # 初始化列名  
    columns = ['file_name'] + ['mode'] + [f'element_({i})' for i in range(-5, 6)] + ['RMSE']  
    
    # 初始化一个空的DataFrame  
    df = pd.DataFrame(columns=columns) 
    return df


def test_csv(df_head, file_path, mode):
 
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

    df_temp = csv_init()
    df_temp.loc[0]=0
    df_temp['file_name'] = file_path
    df_temp["mode"] = str(mode)
    counter, rmse= DT(X, y, X_test, y_test, mode)
    for element, count in counter.items():
        df_temp["element_({})".format(str(element))] = count
    df_temp["RMSE"] = rmse
    df_temp.fillna(0, inplace=True)
    df_head.loc[-1] = df_temp.values[0]
    df_head = df_head.reset_index(drop=True)
    print(df_head)
    df_head.to_csv("test_result.csv", index=False)
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
                 "merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_4.csv"]
    
    int_mode(None)

    # 文件名  
    filename = 'test_result.csv'  
    for file in file_list:
        for mode in mod_deep_learn.keys():
            df_head = pd.DataFrame()
            # 检查文件是否存在  
            if os.path.exists(filename):  
                # 文件存在，读取到DataFrame  
                print("open csv")
                df_head = pd.read_csv(filename)  
            else:
                print("create csv")
                df_head = csv_init()
            print("./Orignal/" + file, "  " + mode)
            test_csv(df_head, "./Orignal/" + file, mode)
            del df_head