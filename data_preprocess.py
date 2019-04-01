# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : data_preprocess.py
# @time  : 2019/3/26
"""
文件说明：数据预处理，提取基本特征
"""
import lightgbm as lgb
import pandas as pd
import os

path = "F:/数据集/1903地铁预测/"
test = pd.read_csv(open(path + '/Metro_testB/testB_submit_2019-01-27.csv',encoding="utf8"))
test_26 = pd.read_csv(open(path + '/Metro_testB/testB_record_2019-01-26.csv',encoding="utf8"))
test_28 = pd.read_csv(open(path + '/Metro_testA/testA_record_2019-01-28.csv',encoding="utf8"))

def get_base_features(df_):
    df = df_.copy()

    # base time
    df['day'] = df['time'].apply(lambda x: int(x[8:10]))
    df['week'] = pd.to_datetime(df['time']).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df.time).dt.weekday >= 5).astype(int)
    df['hour'] = df['time'].apply(lambda x: int(x[11:13]))
    df['minute'] = df['time'].apply(lambda x: int(x[14:15] + '0'))

    # count,sum
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()

    # nunique deviceID闸机编号
    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result = result.merge(tmp, on=['stationID'], how='left')
    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique(). \
        reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # in,out
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']

    #
    result['day_since_first'] = result['day'] - 1
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    return result

data_28 = get_base_features(test_28)
data_26 = get_base_features(test_26)
data = pd.concat([data_28, data_26], axis=0, ignore_index=True)
#加载所有文件
data_list = os.listdir(path+'/Metro_train/')
for i in range(0, len(data_list)):
    if data_list[i].split('.')[-1] == 'csv':
        print(data_list[i], i)
        df = pd.read_csv(open(path+'/Metro_train/' + data_list[i],encoding="utf8"))
        df = get_base_features(df)
        data = pd.concat([data, df], axis=0, ignore_index=True)
    else:
        continue

def get_test_features(test):
    test['week'] = pd.to_datetime(test['startTime']).dt.dayofweek + 1
    test['weekend'] = (pd.to_datetime(test.startTime).dt.weekday >= 5).astype(int)
    test['day'] = test['startTime'].apply(lambda x: int(x[8:10]))
    test['hour'] = test['startTime'].apply(lambda x: int(x[11:13]))
    test['minute'] = test['startTime'].apply(lambda x: int(x[14:15] + '0'))
    test['day_since_first'] = test['day'] - 1
    test = test.drop(['startTime', 'endTime'], axis=1)
    return test

test = get_test_features(test)
data = pd.concat([data,test], axis=0, ignore_index=True)
############################################################
#构造全部枚举值
temp_df = pd.DataFrame({"minute":[],"hour":[],"day":[],"stationID":[]})
for station in range(81):
    print(station)
    for day in range(1,29):
        for hour in range(24):
            temp = pd.DataFrame({"minute":[0,10,20,30,40,50]})
            temp["hour"] = int(hour)
            temp["day"] = int(day)
            temp["stationID"] = int(station)
            temp_df = pd.concat([temp_df,temp],axis=0)
temp_df = temp_df.reset_index(drop=True)
data_min_all = temp_df.merge(data, on=["stationID", "day", "hour", "minute"], how="left")
data_min_all = data_min_all.fillna(0)

#week填充
def week_fill(df):
    if df["day"] in [7,14,21,28]:
        df["week"]=1
        df["weekend"] = 0
    elif df["day"] in [1,8,15,22,29]:
        df["week"]=2
        df["weekend"] = 0
    elif df["day"] in [2,9,16,23,30]:
        df["week"] = 3
        df["weekend"] = 0
    elif df["day"] in [3,10,17,24,31]:
        df["week"] = 4
        df["weekend"] = 0
    elif df["day"] in [4,11,18,25]:
        df["week"] = 5
        df["weekend"] = 0
    elif df["day"] in [5,12,19,26]:
        df["week"] = 6
        df["weekend"] = 1
    elif df["day"] in [6,13,20,27]:
        df["week"] = 7
        df["weekend"] = 1
    return df
data_min_all = data_min_all.apply(week_fill, axis=1)
data_min_all.to_csv("F:/数据集处理/1903地铁预测/train/data_all_b.csv",index=False)
