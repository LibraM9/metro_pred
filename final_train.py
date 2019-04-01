# -*- coding: utf-8 -*-
#@author: limeng
#@file: final_train.py
#@time: 2019/3/30 22:55
"""
文件说明：
"""
import pandas as pd
import os
import numpy as np
import lightgbm as lgb

data = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all_b.csv",encoding="utf8"))
# 保留周末
holiday = [5,6,12,13,19,20,26,27]
data = data[data["day"].isin(holiday)]

def fix_day(d):
    if d==5:
        return 1
    elif d==6:
        return 2
    elif d==12:
        return 3
    elif d ==13:
        return 4
    elif d ==19:
        return 5
    elif d ==20:
        return 6
    elif d ==26:
        return 7
    else:
        return d
data['day'] = data['day'].apply(fix_day)

stat_columns = ['inNums','outNums']

#提取前一天的记录作为特征
def get_refer_day(d):
    if d == 7:
        return 27
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
    if d==1:
        return 5
    elif d==2:
        return 6
    elif d==3:
        return 12
    elif d ==4:
        return 13
    elif d ==5:
        return 19
    elif d ==6:
        return 20
    elif d ==7:
        return 26
    else:
        return d

all_columns = [f for f in data.columns if f not in ['weekend','inNums','outNums']]
### all data
all_data = data[data.day!=27]
all_data['day'] = all_data['day'].apply(recover_day)
X_data = all_data[all_columns]

train = data[data.day <7]
train['day'] = train['day'].apply(recover_day)
X_train = train[all_columns]

valid = data[data.day==7]
valid['day'] = valid['day'].apply(recover_day)
X_valid = valid[all_columns]

test  = data[data.day==27]
X_test = test[all_columns]

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

sub27 = pd.read_csv(open('F:/数据集/1903地铁预测/Metro_testB/testB_submit_2019-01-27.csv', encoding="utf8"))
del sub27['inNums']
del sub27['outNums']
sub27['hour']=pd.to_datetime(sub27['startTime'], format='%Y-%m-%d %H:%M:%S').dt.hour
sub27['minute']=pd.to_datetime(sub27['startTime'], format='%Y-%m-%d %H:%M:%S').dt.minute
sub27=sub27.merge(test[["stationID", "hour", "minute", "inNums", "outNums"]], on = ['stationID', 'hour', 'minute'], how ='left')
submition=sub27[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']]
submition.loc[submition["stationID"]==54,"inNums"]=0
submition.loc[submition["stationID"]==54,"outNums"]=0
submition.to_csv('F:/数据集处理/1903地铁预测/submit/final_sub/sub2_lgb_67.csv',index=False)