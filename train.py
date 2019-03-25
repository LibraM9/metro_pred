# -*- coding: utf-8 -*-
#@author: limeng
#@file: train.py
#@time: 2019/3/25 22:14
"""
文件说明：fb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_in = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/in.csv", encoding='utf8'))
train_out = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/out.csv", encoding='utf8'))

train_in["day"] = pd.to_datetime(train_in['time'],format='%Y-%m-%d %H:%M:%S').dt.day
train_in["hour"] = pd.to_datetime(train_in['time'],format='%Y-%m-%d %H:%M:%S').dt.hour
train_out["day"] = pd.to_datetime(train_out['time'],format='%Y-%m-%d %H:%M:%S').dt.day
train_out["hour"] = pd.to_datetime(train_out['time'],format='%Y-%m-%d %H:%M:%S').dt.hour

workday = [2,3,4,7,8,9,10,11,14,15,16,17,18,21,22,23,24,25,28]
train_in = train_in[train_in["day"].isin(workday)]
train_out = train_out[train_out["day"].isin(workday)]
time_list = train_in["hour"].unique()

train_in_hour = train_in.groupby(["stationID","day","hour"])["inNums"].agg({"inNums_hour":"sum"}).reset_index()
train_out_hour = train_out.groupby(["stationID","day","hour"])["outNums"].agg({"outNums_hour":"sum"}).reset_index()

d29_in = train_in_hour[train_in_hour["day"]==28].reset_index(drop=True)
d29_in["day"] = 29
d29_out = train_out_hour[train_out_hour["day"]==28].reset_index(drop=True)
d29_out["day"] = 29

station_lst = train_in_hour["stationID"].unique()
hour_lst = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

from fbprophet import Prophet
import pystan
df = train_in_hour[(train_in_hour["stationID"]==0)&(train_in_hour["hour"]==6)][["day","inNums_hour"]]
df.columns = ["ds","y"]
model =Prophet()
model.fit(df)