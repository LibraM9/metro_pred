# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : train_lgb_hour.py
# @time  : 2019/3/26
"""
文件说明：按小时切分，lgb训练
"""
import lightgbm as lgb
import pandas as pd
import os
import tqdm
import numpy as np
from scipy.stats import skew
from tsfresh.feature_extraction import feature_calculators as ts
from tsfresh.feature_extraction import extract_features


path = "F:/数据集/1903地铁预测/"

data = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all.csv",encoding="utf8"))

data_agg = {"inNums":"sum",
            "outNums":"sum",
            "nuni_deviceID_of_stationID":"mean",
            "nuni_deviceID_of_stationID_hour":"mean"
            }
# data_hour = data.groupby(['stationID', 'week', 'weekend', 'day','day_since_first', 'hour'],as_index=False)[
#     "inNums","outNums","nuni_deviceID_of_stationID","nuni_deviceID_of_stationID_hour"].agg(data_agg)
data_hour = data.groupby(['stationID', 'week', 'weekend', 'day', 'hour'],as_index=False)[
    "inNums","outNums","nuni_deviceID_of_stationID","nuni_deviceID_of_stationID_hour"].agg(data_agg)

temp_df = pd.DataFrame({"hour":[],"stationID":[],"day":[]})
for station in range(81):
    print(station)
    for day in range(1,30):
        if day in [26,27]:
            continue
        temp = pd.DataFrame({"hour":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]})
        temp["stationID"] = station
        temp["day"] = day
        temp_df = pd.concat([temp_df,temp],axis=0)
temp_df = temp_df.reset_index(drop=True)
data_hour_all = temp_df.merge(data_hour,on=["stationID","day","hour"],how="left")
data_hour_all = data_hour_all.fillna(0)
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
data_hour_all = data_hour_all.apply(week_fill,axis=1)
#剔除1号和周末
data_hour_all = data_hour_all[(data_hour_all.day!=1)]
data_hour_all = data_hour_all[(data_hour_all.day!=5)&(data_hour_all.day!=6)]
data_hour_all = data_hour_all[(data_hour_all.day!=12)&(data_hour_all.day!=13)]
data_hour_all = data_hour_all[(data_hour_all.day!=19)&(data_hour_all.day!=20)]
data_hour_all = data_hour_all[(data_hour_all.day!=26)&(data_hour_all.day!=27)]
data_hour_all = data_hour_all.reset_index(drop=True)

#构造近三天的标签
def get_traditional_label(df, grp_col = ["stationID","hour"], label_cols = ["inNums","outNums"], day_shifts = [1,2,3,4,5]):
    #近三天特征
    for day in day_shifts:
        for label_col in label_cols:
            df[label_col +  '_last_{}'.format(day)] = df.groupby(grp_col)[label_col].shift(day).values # 注意是往后移动,
    df = df.fillna(0)

    #变化的特征
    for fea_col in label_cols:
        df[fea_col + '_mean_3'] = (df[fea_col+"_last_1"].values + df[fea_col + '_last_2'].values + df[fea_col + '_last_3'].values) / 3.0
        df[fea_col + '_diff_1'] = df[fea_col+"_last_1"].values -  df[fea_col + '_last_2'].values
        df[fea_col + '_diff_2'] = df[fea_col + '_last_2'].values -  df[fea_col + '_last_3'].values
        df[fea_col + '_diff_diff'] = df[fea_col + '_diff_1'].values - df[fea_col + '_diff_2'].values
        df[fea_col + '_divide_1'] = df[fea_col+"_last_1"].values / (df[fea_col + '_last_2'].values + 0.001)
        df[fea_col + '_divide_2'] = df[fea_col+"_last_2"].values / (df[fea_col + '_last_3'].values + 0.001)
    return df
data_hour_all = get_traditional_label(data_hour_all)

# data_hour_all_test = data_hour_all[(data_hour_all["stationID"]==0)&(data_hour_all["hour"]==0)]
# data_hour_all_test["test"]=data_hour_all_test.groupby(["stationID","hour"])["inNums"].shift(1)

#过去N天的统计特征
def get_sts_features(df):
    for fea_col in ["inNums","outNums"]:
        for slide_windows in [3,5]:
            print(fea_col, slide_windows)
            slide_cols = [fea_col + '_last_' + str(i + 1) for i in range(slide_windows)]
            df_tmp = df[slide_cols].values
            df_tmp_percent = df[slide_cols].copy()
            df_tmp_percent['sum_'] = df_tmp_percent.sum(axis=1).values
            for col in slide_cols:
                df_tmp_percent[col] = df_tmp_percent[col].values / (0.001 + df_tmp_percent['sum_'].values)  # 百分比
            df_tmp_percent = df_tmp_percent[slide_cols].values
            # df_grp = df.groupby(["stationID","hour"])[slide_cols].sum(axis=1).reset_index()
            # df_city_dic = df.groupby(["stationID","hour"])[slide_cols].sum(axis=1).to_dict()
            df['district_' + fea_col + '_last{}_sum'.format(slide_windows)] = np.sum(df_tmp, axis=1)
            df['district_' + fea_col + '_last{}_median'.format(slide_windows)] = np.median(df_tmp, axis=1)
            df['district_' + fea_col + '_last{}_std'.format(slide_windows)] = np.std(df_tmp, axis=1)
            df['district_' + fea_col + '_last{}_min'.format(slide_windows)] = np.min(df_tmp, axis=1)
            df['district_' + fea_col + '_last{}_max'.format(slide_windows)] = np.max(df_tmp, axis=1)
            df['district_' + fea_col + '_last{}_mean_change'.format(slide_windows)] = df[slide_cols].apply(ts.mean_change, axis=1)

            df['district_percent_' + fea_col + '_last{}_median'.format(slide_windows)] = np.median(df_tmp_percent,axis=1)
            df['district_percent_' + fea_col + '_last{}_std'.format(slide_windows)] = np.std(df_tmp_percent, axis=1)
            df['district_percent_' + fea_col + '_last{}_min'.format(slide_windows)] = np.min(df_tmp_percent, axis=1)
            df['district_percent_' + fea_col + '_last{}_max'.format(slide_windows)] = np.max(df_tmp_percent, axis=1)
            df['district_percent_' + fea_col + '_last{}_skew'.format(slide_windows)] = skew(df_tmp, axis=1)
    return df

data_hour_all = get_sts_features(data_hour_all)
#按week,hour,minute分别对inNums和outNums构造统计特征
tmp = data_hour.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_wh_max'    : 'max',
                                                                        'inNums_wh_min'    : 'min',
                                                                        'inNums_wh_mean'   : 'mean',
                                                                        'inNums_wh_std': 'std'
                                                                        })
data_hour_all = data_hour_all.merge(tmp, on=['stationID','week','hour'], how='left')

tmp = data_hour.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                        'outNums_wh_max'    : 'max',
                                                                        'outNums_wh_min'    : 'min',
                                                                        'outNums_wh_mean'   : 'mean',
                                                                        'outNums_wh_std'   : 'std'
                                                                        })
data_hour_all = data_hour_all.merge(tmp, on=['stationID','week','hour'], how='left')
data_hour_all = data_hour_all.fillna(0)
#去除前3天数据
data_hour_all = data_hour_all[(data_hour_all.day!=2)&(data_hour_all.day!=3)&(data_hour_all.day!=4)]
all_columns = [f for f in data_hour_all.columns if f not in ['weekend','inNums','outNums'
    ,"nuni_deviceID_of_stationID","nuni_deviceID_of_stationID_hour"]]

# all data
all_data = data_hour_all[data_hour_all.day!=29]
X_data = all_data[all_columns]

train = data_hour_all[data_hour_all.day <28]
X_train = train[all_columns]

valid = data_hour_all[data_hour_all.day==28]
X_valid = valid[all_columns]

test  = data_hour_all[data_hour_all.day==29]
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
                num_boost_round=20000,
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
                num_boost_round=20000,
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
test[["stationID","hour","inNums","outNums"]].to_csv("F:/数据集处理/1903地铁预测/train/lgb_hour_pre29_0327.csv",index=False)
