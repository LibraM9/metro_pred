# -*- coding: utf-8 -*-
#@author: limeng
#@file: train_fb.py
#@time: 2019/3/25 22:14
"""
文件说明：fb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = "F:/数据集/1903地铁预测/"
data = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all.csv",encoding="utf8"))
#转化为小时级别
data_agg = {"inNums":"sum",
            "outNums":"sum",
            "nuni_deviceID_of_stationID":"mean",
            "nuni_deviceID_of_stationID_hour":"mean"
            }
data_hour = data.groupby(['stationID', 'week', 'weekend', 'day','day_since_first', 'hour'],as_index=False)[
    "inNums","outNums","nuni_deviceID_of_stationID","nuni_deviceID_of_stationID_hour"].agg(data_agg)

#构造枚举值
temp_df = pd.DataFrame({"hour":[],"stationID":[],"day":[]})
for station in range(81):
    print(station)
    for day in range(1,30):
        if day in [26,27]:
            continue
        temp = pd.DataFrame({"hour":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]})
        temp["stationID"] = station
        temp["day"] = day
        temp_df = pd.concat([temp_df,temp],axis=0)
temp_df = temp_df.reset_index()
data_hour_all = temp_df.merge(data_hour,on=["stationID","day","hour"],how="left")
data_hour_all = data_hour_all.fillna(0)

#剔除周末
data_hour_all = data_hour_all[(data_hour_all.day!=1)]
data_hour_all = data_hour_all[(data_hour_all.day!=5)&(data_hour_all.day!=6)]
data_hour_all = data_hour_all[(data_hour_all.day!=12)&(data_hour_all.day!=13)]
data_hour_all = data_hour_all[(data_hour_all.day!=19)&(data_hour_all.day!=20)]
data_hour_all = data_hour_all[(data_hour_all.day!=26)&(data_hour_all.day!=27)]

test = data_hour_all[(data_hour_all.day==29)]
data_hour_all = data_hour_all[(data_hour_all.day!=29)]
# data_hour_all_test = data_hour_all[(data_hour_all.day!=28)&(data_hour_all.day!=29)]
from fbprophet import Prophet
import numpy as np
#inNums
for station in range(81):
    print(station)
    for hour in range(24):
        df = data_hour_all[(data_hour_all["stationID"]==station)&(data_hour_all["hour"]==hour)][["day","inNums"]]
        df.columns = ["ds","y"]
        model =Prophet()
        model.fit(df)
        forecast =model.predict(pd.DataFrame({"ds":[29]}))
        test.loc[(test["stationID"]==station)&(test["hour"]==hour),"inNums"]=forecast["trend"].values[0]

#outNums
for station in range(81):
    print(station)
    for hour in range(24):
        df = data_hour_all[(data_hour_all["stationID"]==station)&(data_hour_all["hour"]==hour)][["day","outNums"]]
        df.columns = ["ds","y"]
        model =Prophet()
        model.fit(df)
        forecast =model.predict(pd.DataFrame({"ds":[29]}))
        test.loc[(test["stationID"]==station)&(test["hour"]==hour),"outNums"]=forecast["trend"].values[0]


test[["stationID","hour","inNums","outNums"]].to_csv("F:/数据集处理/1903地铁预测/train/fb_hour_pre29.csv",index=False)
