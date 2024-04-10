import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np

# 基类
class SamplingMethod:
    def __init__(self) -> None:
        pass
    def fit_resample(self, X, y):
        raise NotImplementedError("Subclasses must implement fit_resample method.")
    def sample_name(self):
        return 'None'

# SMOTE 类的一个实现
class SamplingMethod_SMOTE(SamplingMethod):
    def __init__(self) -> None:
        super().__init__()
        self.sampling_method = SMOTE()
    def fit_resample(self, X, y):
        # 实现 SMOTE 的过采样逻辑
        return self.sampling_method.fit_resample(X, y)
    def sample_name(self):
        return 'SMOTE'

# 另一个采样方法的类的实现
class AnotherSamplingMethod(SamplingMethod):
    def fit_resample(self, X, y):
        # 实现另一个采样方法的逻辑
        pass

class IPdMData(SMOTE):
    def __init__(self, sampling_method: SamplingMethod, file_path:str, files_name:list) -> None:
        self.sampling_method = sampling_method
        self.dataList = []
        self.new_file_name = 'merged_'
        memory_usage = 0
        for file_name in files_name:
            print("reading data from ", file_name)
            self.new_file_name += file_name[:-4]
            temp_data = pd.read_csv(file_path + file_name)

            memory_usage += temp_data.memory_usage(deep=True).sum()

            # 统一时间格式
            # 将日期时间字符串转换为 datetime 对象，尝试自动推断日期时间格式
            if 'datetime' in temp_data.columns:
                temp_data['datetime'] = pd.to_datetime(temp_data['datetime'])
                # 格式化日期时间为指定格式 "%Y/%m/%d %H:%M"
                temp_data['datetime'] = temp_data['datetime'].dt.strftime("%Y/%m/%d %H:%M")
            self.dataList.append(temp_data)
        print("Data loading completed. data size : ", int(memory_usage / 1024), "KB")
        print("**********************************************************")
    
    def mergeDataFiles(self, step_size:int, is_save:bool):

        # 合并表格  
        data_telemetry_errors = pd.merge(self.dataList[2], self.dataList[0], on=["datetime", "machineID"], how="outer")
        # 检验是否errorID和PdM_errors.csv中是否一致
        # print(data_telemetry_errors[data_telemetry_errors['errorID'] == "error1"])
        data_telemetry_errors_machines = pd.merge(data_telemetry_errors, self.dataList[1], on=["machineID"])

        # 替换errorID和modelPdM_failures.csv
        error_mapping = {
            'error1': 1,
            'error2': 2,
            'error3': 3,
            'error4': 4,
            'error5': 5
        }
        data_telemetry_errors_machines['errorID'] = data_telemetry_errors_machines['errorID'].map(error_mapping)
        data_telemetry_errors_machines['errorID'] = data_telemetry_errors_machines['errorID'].fillna(value=int(0))
        data_telemetry_errors_machines['errorID'] = data_telemetry_errors_machines['errorID'].astype('int64')


        model_mapping = {
            'model1' : 1,
            'model2' : 2,
            'model3' : 3,
            'model4' : 4
        }
        data_telemetry_errors_machines['model'] = data_telemetry_errors_machines['model'].map(model_mapping)
        data_telemetry_errors_machines['model'] = data_telemetry_errors_machines['model'].astype('int64')

        df = data_telemetry_errors_machines

        # time shift新表格，步长step_size
        df_machineID_list = list(df['machineID'].unique())
        for i in df_machineID_list:
            df_machineID = df[df['machineID'] == i].copy()
            df_machineID.loc[df_machineID.index, 'errorID'] = df_machineID['errorID'].shift(-step_size)
            df[df['machineID'] == i] = df_machineID.copy()
    
        # 去除time shift后，最后两行中errorID为空的两行
        df = df.dropna()
        df = df.reset_index()

        self.merged_data = df

        num_rows, num_cols = self.merged_data.shape
        print("Data merging completed.", "rows : ", num_rows, "cols : ", num_cols)

        # 保存表格
        self.new_file_name += '_TSL_' + str(step_size)
        if is_save:
            df.to_csv(self.new_file_name  + '.csv', index=False)
            print("Successfully saved the merged data. file name : ", self.new_file_name + ".csv")
        print("**********************************************************")
    
        return df

    def balanceData(self, is_save:bool=False, isdebug:bool=False):
        df = self.merged_data.copy()
        df["datetime"] = pd.to_datetime(df['datetime'], format="%Y/%m/%d %H:%M")
        df["datetime"] = df["datetime"].apply(lambda x:time.mktime(x.timetuple()))
        df['errorID'] = df['errorID'].astype('int64')
        
        y = df['errorID']
        x = df.drop('errorID', axis=1)
        if isdebug == True:
            groupby_data_original = df.groupby("errorID").count()
            print(groupby_data_original)
        x_smote_resampled, y_smote_resampled = self.sampling_method.fit_resample(x, y)
        x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=['index', 'datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration', 'model', 'age'])
        y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=["errorID"])

        self.balancedData = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)

        print("The data has been successfully balanced using the ", self.sampling_method.sample_name(), " method.")
        if isdebug == True:
            groupby_data_smote = self.balancedData.groupby("errorID").count()
            print("**********************************************************")
            print(groupby_data_smote)
        self.new_file_name += '_balance_' + type(self).__bases__[0].__name__
        # 保存表格
        if is_save:
            self.balancedData.to_csv(self.new_file_name + '.csv', index=False)
            print("Successfully saved the balanced data. file name : ", self.new_file_name + ".csv")
        print("**********************************************************")
        
        return self.balancedData

# e.g.
if __name__ == '__main__':
    sample_file = ["PdM_errors.csv", "PdM_machines.csv", "PdM_telemetry.csv"]
    filePath = "./"
    imb_sampling = SamplingMethod_SMOTE()
    pdmData = IPdMData(imb_sampling, filePath, sample_file)
    pdmData.mergeDataFiles(2, True)
    pdmData.balanceData(True)

    # df = pd.read_csv("merged_PdM_errorsPdM_machinesPdM_telemetry_TSL_2.csv")
    # df = df.head(300)
    # df["datetime"] = pd.to_datetime(df['datetime'], format="%Y/%m/%d %H:%M")
    # df["datetime"] = df["datetime"].apply(lambda x:time.mktime(x.timetuple()))
    # df['errorID'] = df['errorID'].astype('int64')
    # # 根据 error ID 筛选数据
    # data = df
    # pressure_mean = data['pressure'].mean()
    # data['pressure'] = data['pressure'] -pressure_mean
    # error_0_data = data[data['errorID'] == 0]
    # error_not_0_data = data[data['errorID'] != 0]

    # # 创建两个子图
    # fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # # 第一张图：error ID 是 0 的情况
    # axes[0].plot(error_0_data['datetime'], error_0_data['pressure'], color='blue')
    # axes[0].set_title('Pressure vs. Datetime (Error ID = 0)')
    # axes[0].set_xlabel('Datetime')
    # axes[0].set_ylabel('Pressure')

    # # 第二张图：error ID 不等于 0 的情况
    # axes[1].plot(error_not_0_data['datetime'], error_not_0_data['pressure'], color='red')
    # axes[1].set_title('Pressure vs. Datetime (Error ID ≠ 0)')
    # axes[1].set_xlabel('Datetime')
    # axes[1].set_ylabel('Pressure')

    # # 调整子图之间的间距
    # plt.tight_layout()

    # # 显示图形
    # plt.show()
    
    # # 对 pressure 数据进行离散傅里叶变换
    # def calculate_fft(data):
    #     pressure_fft = fft(data['pressure'].to_numpy())
    #     n = len(data)
    #     freq = np.fft.fftfreq(n)
    #     return freq[:n//2], np.abs(pressure_fft)[:n//2]  # 仅取正频率部分
    # # 计算离散傅里叶变换并获取频率和幅度
    # freq_0, amplitude_0 = calculate_fft(error_0_data)
    # freq_not_0, amplitude_not_0 = calculate_fft(error_not_0_data)

    # # 绘制频谱图
    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # plt.scatter(freq_0, amplitude_0, color='blue')
    # plt.title('Frequency Spectrum (Error ID = 0)')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # # 绘制竖线
    # plt.vlines(freq_0, ymin=0, ymax=amplitude_0, color='red', linestyle='--')

    # plt.subplot(2, 1, 2)
    # plt.scatter(freq_not_0, amplitude_not_0, color='red', s=5)
    # plt.title('Frequency Spectrum (Error ID ≠ 0)')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # # 绘制竖线
    # plt.vlines(freq_not_0, ymin=0, ymax=amplitude_not_0, color='red', linestyle='--')

    # plt.tight_layout()
    # plt.show()