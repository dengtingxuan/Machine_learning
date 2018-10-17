"""This code is for experimenting to build a model of predicting window opening behaviors"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

reside=["J3","Z4"]
for res in reside:
    data=pd.read_csv("F:/201810_ML_Learning/{}_annually_data_Handled.csv".format(res))
    print(data.columns)
    x=data[['Hour', 'Month', 'Week', 'Temperature', 'RH', 'HCHO', 'CO2', 'PM2.5','Out_Temperature',
            'Out_RH', 'Rain', 'CO(mg/m3)', 'NO2(ug/m3)','SO2(ug/m3)', 'O3(ug/m3)', 'PM10(ug/m3)',
            'PM2.5(ug/m3)', 'AQI']]
    y=data['BRW']
    y_n=pd.DataFrame()
    a0=pd.DataFrame({"a":[1],"b":[0]},columns=["a","b"])
    a1=pd.DataFrame({"a": [0], "b": [1]}, columns=["a", "b"])

    print(x.shape)
    print(y.shape)

    for i in range(len(y)):
        if y.iloc[i]==1:
            y_n= y_n.append(a0)
        else:
            y_n = y_n.append(a1)
        print(i)
    print(y_n)
    y_n.to_csv("F:/201810_ML_Learning/Tensorflow/{}_annually_data_Handled_For_TF_win.csv".format(res))
    y_n=pd.read_csv("F:/201810_ML_Learning/Tensorflow/{}_annually_data_Handled_For_TF_win.csv".format(res))
    data_n=pd.concat((x,y_n),axis=1)
    print(data_n)
    data_n.to_csv("F:/201810_ML_Learning/Tensorflow/{}_annually_data_Handled_For_TF_Compleled.csv".format(res))