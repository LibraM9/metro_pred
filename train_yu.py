# -*- coding: utf-8 -*-
#@author: limeng
#@file: train_yu.py
#@time: 2019/3/25 23:42
"""
文件说明：预处理
https://zhuanlan.zhihu.com/p/59998657
13.7446
"""
import pandas as pd
import os
import numpy as np
import lightgbm as lgb

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

    # nunique
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

# 剔除周末,并修改为连续时间
data = data[(data.day!=5)&(data.day!=6)]
data = data[(data.day!=12)&(data.day!=13)]
data = data[(data.day!=19)&(data.day!=20)]
data = data[(data.day!=26)&(data.day!=27)]

def fix_day(d):
    if d in [1,2,3,4]:
        return d
    elif d in [7,8,9,10,11]:
        return d - 2
    elif d in [14,15,16,17,18]:
        return d - 4
    elif d in [21,22,23,24,25]:
        return d - 6
    elif d in [28]:
        return d - 8
data['day'] = data['day'].apply(fix_day)

#拼接测试集
test['week']    = pd.to_datetime(test['startTime']).dt.dayofweek + 1
test['weekend'] = (pd.to_datetime(test.startTime).dt.weekday >=5).astype(int)
test['day']     = test['startTime'].apply(lambda x: int(x[8:10]))
test['hour']    = test['startTime'].apply(lambda x: int(x[11:13]))
test['minute']  = test['startTime'].apply(lambda x: int(x[14:15]+'0'))
test['day_since_first'] = test['day'] - 1
test = test.drop(['startTime','endTime'], axis=1)
data = pd.concat([data,test], axis=0, ignore_index=True)

stat_columns = ['inNums','outNums']

#提取前一天的记录作为特征 todo 使用其他方法提取前一天数据
def get_refer_day(d):
    if d == 20:
        return 29
    else:
        return d + 1

tmp = data.copy()
tmp_df = tmp[tmp.day == 1]
tmp_df['day'] = tmp_df['day'] - 1
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
tmp['day'] = tmp['day'].apply(get_refer_day)

for f in stat_columns:
    tmp.rename(columns={f: f + '_last'}, inplace=True)

tmp = tmp[['stationID', 'day', 'hour', 'minute', 'inNums_last', 'outNums_last']]

data = data.merge(tmp, on=['stationID', 'day', 'hour', 'minute'], how='left')
data.fillna(0, inplace=True)

#按week,hour,minute分别对inNums和outNums构造统计特征
tmp = data.groupby(['stationID','week','hour','minute'], as_index=False)['inNums'].agg({
                                                                        'inNums_whm_max'    : 'max',
                                                                        'inNums_whm_min'    : 'min',
                                                                        'inNums_whm_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour','minute'], how='left')

tmp = data.groupby(['stationID','week','hour','minute'], as_index=False)['outNums'].agg({
                                                                        'outNums_whm_max'    : 'max',
                                                                        'outNums_whm_min'    : 'min',
                                                                        'outNums_whm_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour','minute'], how='left')

tmp = data.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_wh_max'    : 'max',
                                                                        'inNums_wh_min'    : 'min',
                                                                        'inNums_wh_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour'], how='left')

tmp = data.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                        #'outNums_wh_max'    : 'max',
                                                                        #'outNums_wh_min'    : 'min',
                                                                        'outNums_wh_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour'], how='left')

#恢复初始时间
def recover_day(d):
    if d in [1,2,3,4]:
        return d
    elif d in [5,6,7,8,9]:
        return d + 2
    elif d in [10,11,12,13,14]:
        return d + 4
    elif d in [15,16,17,18,19]:
        return d + 6
    elif d == 20:
        return d + 8
    else:
        return d

#加入外部数据
station_detail = pd.read_excel("F:/代码空间/1903地铁预测/station_id.xlsx")
data_min_all_temp = data.merge(station_detail,on="stationID",how="left")
def is_run(df):
    hour = str(int(df["hour"]))
    if len(hour)==1:
        hour = "0"+hour
    minute = str(int(df["minute"]))
    if len(minute)==1:
        minute = "0"+minute
    temp_time = hour+":"+minute
    if len(temp_time)==4:
        temp_time = "0"+temp_time
    if (temp_time<min(df["start_time"],df["start_time2"])) | (temp_time>max(df["end_time"],df["end_time2"])):
        return 0
    else:
        return 1
data_min_all_temp["is_run"] = data_min_all_temp.apply(is_run, axis=1)
del data_min_all_temp["start_time"],data_min_all_temp["start_time2"]
del data_min_all_temp["end_time"],data_min_all_temp["end_time2"]
data = data_min_all_temp
#去除特殊站
data = data[data["stationID"].isin([7,15,54]).apply(lambda x:bool(1-x))]
#变量筛选
all_columns = [f for f in data.columns if f not in ['weekend','inNums','outNums']]
### all data
all_data = data[data.day!=29]
all_data['day'] = all_data['day'].apply(recover_day)
all_data = all_data[all_data.day!=1] #去除第一天
X_data = all_data[all_columns].values

train = data[data.day <20]
train['day'] = train['day'].apply(recover_day)
train = train[train.day!=1] #去除第一天
X_train = train[all_columns].values

valid = data[data.day==20]
valid['day'] = valid['day'].apply(recover_day)
X_valid = valid[all_columns].values

test  = data[data.day==29]
X_test = test[all_columns].values

#构建模型并训练
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':1,
    'reg_lambda':2
}

y_train = train['inNums']
y_valid = valid['inNums']
y_data  = all_data['inNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train,lgb_evals],
                valid_names=['train','valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )

### all_data
lgb_train = lgb.Dataset(X_data, y_data)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=gbm.best_iteration,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=1000,
                )
test['inNums'] = gbm.predict(X_test)

######################################################outNums
y_train = train['outNums']
y_valid = valid['outNums']
y_data  = all_data['outNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train,lgb_evals],
                valid_names=['train','valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )

### all_data
lgb_train = lgb.Dataset(X_data, y_data)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=gbm.best_iteration,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=1000,
                )
test['outNums'] = gbm.predict(X_test)

test["inNums"] = test["inNums"].apply(lambda x:x if x>=0 else 0)
test["outNums"] = test["outNums"].apply(lambda x:x if x>=0 else 0)

#修复7，15，54
rule = pd.read_csv('F:/数据集处理/1903地铁预测/submit/sub_rulebili_lgbmin_stack.csv')
rule['hour']=pd.to_datetime(rule['startTime'],format='%Y-%m-%d %H:%M:%S').dt.hour
rule['minute']=pd.to_datetime(rule['startTime'],format='%Y-%m-%d %H:%M:%S').dt.minute
test = test[["stationID","hour","minute","inNums","outNums"]]
test = pd.concat([test,rule.loc[rule["stationID"].isin([7,15,54]),["stationID","hour","minute","inNums","outNums"]]],axis=0)
#导出结果
sub = pd.read_csv(open(path + '/Metro_testA/testA_submit_2019-01-29.csv',encoding="utf8"))
del sub['inNums']
del sub['outNums']
sub['hour']=pd.to_datetime(sub['startTime'],format='%Y-%m-%d %H:%M:%S').dt.hour
sub['minute']=pd.to_datetime(sub['startTime'],format='%Y-%m-%d %H:%M:%S').dt.minute
sub=sub.merge(test,on = ['stationID', 'hour', 'minute'],how = 'left')
submition=sub[['stationID','startTime','endTime','inNums','outNums']]
submition.loc[submition["stationID"]==54,"inNums"]=0
submition.loc[submition["stationID"]==54,"outNums"]=0

submition[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('F:/数据集处理/1903地铁预测/submit/lgb_min_yu_pro.csv',index=False)
