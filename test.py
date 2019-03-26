# -*- coding: utf-8 -*-
# @author: limeng
# @file: test.py
# @time: 2019/3/23 23:39
"""
文件说明：
"""
import pandas as pd

sub = pd.read_csv(open("F:/数据集处理/1903地铁预测/submit/submits102.csv", encoding="utf8"))
sub['hours'] = pd.to_datetime(sub['startTime'], format='%Y-%m-%d %H:%M:%S').dt.hour
sub["inNums"] = sub["inNums"] * 0.965 / 1.02
sub["outNums"] = sub["outNums"] * 0.965 / 1.02

sub_new = sub.copy()
work_time = sub[(sub["hours"].isin([6,7, 8, 9,10,11,12,13,14,15,16,17, 18, 19,20,21,22,23]))&(sub["stationID"]==7)]
work_time["outNums"] = work_time["outNums"] * 1.05

for i in work_time.index:
    sub_new["inNums"][i] = work_time["inNums"][i]
    sub_new["outNums"][i] = work_time["outNums"][i]
del sub_new["hours"]
sub_new.to_csv("F:/数据集处理/1903地铁预测/submit/submits0965_s7.csv",index=False)

sub.groupby("stationID")[["inNums"]].shift(2)
