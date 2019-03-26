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
test = pd.read_csv(open(path + '/Metro_testA/testA_submit_2019-01-29.csv',encoding="utf8"))
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

data = get_base_features(test_28)

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
data.to_csv("F:/数据集处理/1903地铁预测/train/data_all.csv",index=False)
