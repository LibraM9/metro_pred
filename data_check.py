# -*- coding: utf-8 -*-
# @author: limeng
# @file: data_check.py
# @time: 2019/3/21 23:25
"""
文件说明：数据探查
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# train_ori1 = pd.read_csv(open("F:/数据集/1903地铁预测/Metro_train/record_2019-01-01.csv",encoding='utf8'))
# train_ori2= pd.read_csv(open("F:/数据集/1903地铁预测/Metro_train/record_2019-01-02.csv",encoding='utf8'))
# train_in = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/in.csv", encoding='utf8'))
# train_out = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/out.csv", encoding='utf8'))
#
# train_in["date"] = train_in["time"].apply(lambda x: int(x.split(" ")[0][8:]))
# train_in["time"] = train_in["time"].apply(lambda x: x.split(" ")[1])
# train_out["date"] = train_out["time"].apply(lambda x: int(x.split(" ")[0][8:]))
# train_out["time"] = train_out["time"].apply(lambda x: x.split(" ")[1])

data = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all_b.csv",encoding="utf8"))

def time_trans(df):
    hour = str(int(df["hour"]))
    if len(hour)==1:
        hour = "0"+hour
    minute = str(int(df["minute"]))
    if len(minute)==1:
        minute = "0"+minute
    temp_time = hour+":"+minute
    return temp_time
train_in = data[["stationID","day","hour","minute","inNums"]]
train_in["time"] = train_in.apply(time_trans,axis=1)
train_out = data[["stationID","day","hour","minute","outNums"]]
train_out["time"] = train_out.apply(time_trans,axis=1)
#筛选非休息日
# workday = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
workday = [1,5,6,12,13,19,20,26,27,28]
train_in = train_in[train_in["day"].isin(workday)]
train_out = train_out[train_out["day"].isin(workday)]
time_list = sorted(train_in["time"].unique())

def station_plot(station):
    n_fig = 12
    time_list_sub = time_list[:]

    f, ax = plt.subplots(n_fig, n_fig, figsize=(120, 90))

    for i in range(len(time_list_sub)):
        try:
            time = time_list_sub[i]
            temp_in = train_in[(train_in["stationID"] == station) & (train_in["time"] == time)].sort_values(by="day")
            temp_in = temp_in.set_index("day")
            temp_out = train_out[(train_out["stationID"] == station) & (train_out["time"] == time)].sort_values(by="day")
            temp_out = temp_out.set_index("day")

            num = int(i / n_fig)
            temp_in["inNums"].plot.line(color='b',marker="o",ax=ax[num, i - num * n_fig],legend="in",title=str(station)+" station->"+time)
            temp_out["outNums"].plot.line(color='r', marker="+",ax=ax[num, i - num * n_fig],legend="out")
            # temp_in["inNums"].plot.line(color='b', legend="in",title=str(station) + " station->" + time)
            # temp_out["outNums"].plot.line(color='r', legend="out")
            # plt.show()
        except:
            continue
    plt.savefig("F:/数据集处理/1903地铁预测/fig_holiday_min/{}.jpg".format(str(station)))

files = os.listdir("F:/数据集处理/1903地铁预测/fig_holiday_min")
for i in train_in["stationID"].unique():
    if str(i)+".jpg" not in files:
        print(i)
        station_plot(i)