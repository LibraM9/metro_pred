# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : train_lgb.py
# @time  : 2019/3/26
"""
文件说明：按小时切分，lgb训练
"""
import lightgbm as lgb
import pandas as pd
import os

path = "F:/数据集/1903地铁预测/"

data = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all.csv",encoding="utf8"))

data_agg = {"inNums":"sum",
            "outNums":"sum",
            "nuni_deviceID_of_stationID":"mean",
            "nuni_deviceID_of_stationID_hour":"mean"
            }
data_hour = data.groupby(['stationID', 'week', 'weekend', 'day','day_since_first', 'hour'],as_index=False)[
    "inNums","outNums","nuni_deviceID_of_stationID","nuni_deviceID_of_stationID_hour"].agg(data_agg)

#剔除1号和周末
data_hour = data_hour[(data_hour.day!=1)]
data_hour = data_hour[(data_hour.day!=5)&(data_hour.day!=6)]
data_hour = data_hour[(data_hour.day!=12)&(data_hour.day!=13)]
data_hour = data_hour[(data_hour.day!=19)&(data_hour.day!=20)]
data_hour = data_hour[(data_hour.day!=26)&(data_hour.day!=27)]

def fix_day(d):
    if d in [2,3,4]:
        return d - 1
    elif d in [7,8,9,10,11]:
        return d - 3
    elif d in [14,15,16,17,18]:
        return d - 5
    elif d in [21,22,23,24,25]:
        return d - 7
    elif d in [28]:
        return d - 9
    elif d in [29]:
        return d

data_hour['day'] = data_hour['day'].apply(fix_day)

#提取前一天特征，将第一天的前一天设为自己
stat_columns = ['inNums','outNums']

def get_refer_day(d):
    if d == 19:
        return 29
    else:
        return d + 1

tmp = data_hour.copy()
tmp_df = tmp[tmp.day == 1]
tmp_df['day'] = tmp_df['day'] - 1
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
tmp['day'] = tmp['day'].apply(get_refer_day)

for f in stat_columns:
    tmp.rename(columns={f: f + '_last1'}, inplace=True)

tmp = tmp[['stationID', 'day', 'hour', 'inNums_last1', 'outNums_last1']]

data_hour = data_hour.merge(tmp, on=['stationID', 'day', 'hour'], how='left')
data_hour.fillna(0, inplace=True)

#按week,hour,minute分别对inNums和outNums构造统计特征
tmp = data_hour.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_wh_max'    : 'max',
                                                                        'inNums_wh_min'    : 'min',
                                                                        'inNums_wh_mean'   : 'mean'
                                                                        })
data_hour = data_hour.merge(tmp, on=['stationID','week','hour'], how='left')

tmp = data_hour.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                        #'outNums_wh_max'    : 'max',
                                                                        #'outNums_wh_min'    : 'min',
                                                                        'outNums_wh_mean'   : 'mean'
                                                                        })
data_hour = data_hour.merge(tmp, on=['stationID','week','hour'], how='left')

#恢复初始时间
def recover_day(d):
    if d in [1,2,3]:
        return d+1
    elif d in [4,5,6,7,8]:
        return d + 3
    elif d in [9,10,11,12,13]:
        return d + 5
    elif d in [14,15,16,17,18]:
        return d + 7
    elif d == 19:
        return d + 9
    else:
        return d

all_columns = [f for f in data_hour.columns if f not in ['weekend','inNums','outNums']]
### all data
all_data = data_hour[data_hour.day!=29]
all_data['day'] = all_data['day'].apply(recover_day)
X_data = all_data[all_columns].values

train = data_hour[data_hour.day <19]
train['day'] = train['day'].apply(recover_day)
X_train = train[all_columns].values

valid = data_hour[data_hour.day==19]
valid['day'] = valid['day'].apply(recover_day)
X_valid = valid[all_columns].values

test  = data_hour[data_hour.day==29]
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
                num_boost_round=gbm.best_iteration, #先带验证集训练，寻找一个最优迭代次数，然后全数据训练
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

test[["stationID","hour","inNums","outNums"]].to_csv("F:/数据集处理/1903地铁预测/train/lgb_hour_pre29.csv",index=False)
