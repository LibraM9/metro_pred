# -*- coding: utf-8 -*-
#@author: limeng
#@file: data_check_hour.py
#@time: 2019/3/25 21:12
"""
文件说明：以小时为单位进行画图 排除周末
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

def station_plot(station):
    n_fig_x = 4
    n_fig_y = 6
    time_list_sub = time_list[:]

    f, ax = plt.subplots(n_fig_x, n_fig_y, figsize=(80, 40))

    for i in range(len(time_list_sub)):
        time = time_list_sub[i]
        temp_in = train_in_hour[(train_in_hour["stationID"] == station) & (train_in_hour["hour"] == time)].sort_values(by="day")
        temp_in = temp_in.set_index("day")
        temp_out = train_out_hour[(train_out_hour["stationID"] == station) & (train_out_hour["hour"] == time)].sort_values(by="day")
        temp_out = temp_out.set_index("day")

        num = int(i / n_fig_y)
        temp_in["inNums_hour"].plot.line(color='b',marker="o",ax=ax[num, i - num * n_fig_y],legend="in",title=str(station)+" station->"+str(time))
        temp_out["outNums_hour"].plot.line(color='r', marker="+",ax=ax[num, i - num * n_fig_y],legend="out")
        # temp_in["inNums"].plot.line(color='b', legend="in",title=str(station) + " station->" + time)
        # temp_out["outNums"].plot.line(color='r', legend="out")
        # plt.show()
    plt.savefig("F:/数据集处理/1903地铁预测/fig_work_hour/{}.png".format(str(station)))

files = os.listdir("F:/数据集处理/1903地铁预测/fig_work_hour")
for i in train_in["stationID"].unique():
    if str(i)+".png" not in files:
        print(i)
        station_plot(i)