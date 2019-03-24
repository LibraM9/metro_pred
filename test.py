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

work_time = sub[sub["hours"].isin([7, 8, 9, 18, 19])]
unwork_time = sub[sub["hours"].isin([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23])]
unwork_time["inNums"] = unwork_time["inNums"] * 0.98
unwork_time["outNums"] = unwork_time["outNums"] * 0.98

sub_new = pd.concat([work_time, unwork_time])
sub_new = sub_new.sort_index()
del sub_new["hours"]
sub_new.to_csv("F:/数据集处理/1903地铁预测/submit/submits0965_098.csv",index=False)
