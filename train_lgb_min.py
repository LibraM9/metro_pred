# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : train_lgb_min.py
# @time  : 2019/3/27
"""
文件说明：按分钟训练
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

#构造全部枚举值
temp_df = pd.DataFrame({"minute":[],"hour":[],"day":[],"stationID":[]})
for station in range(81):
    print(station)
    for day in range(1,30):
        if day in [26,27]:
            continue
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

#剔除1号和周末
data_min_all = data_min_all[(data_min_all.day != 1)]
data_min_all = data_min_all[(data_min_all.day != 5) & (data_min_all.day != 6)]
data_min_all = data_min_all[(data_min_all.day != 12) & (data_min_all.day != 13)]
data_min_all = data_min_all[(data_min_all.day != 19) & (data_min_all.day != 20)]
data_min_all = data_min_all[(data_min_all.day != 26) & (data_min_all.day != 27)]
data_min_all = data_min_all.reset_index(drop=True)

#构造近三天的标签
def get_traditional_label(df, grp_col = ["stationID","hour","minute"], label_cols = ["inNums","outNums"], day_shifts = [1,2,3,4,5]):
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
data_min_all = get_traditional_label(data_min_all)

# data_hour_all_test = data_hour_all[(data_hour_all["stationID"]==0)&(data_hour_all["hour"]==8)&(data_hour_all["minute"]==20)]
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

data_min_all = get_sts_features(data_min_all)

#按week,hour,minute分别对inNums和outNums构造统计特征
tmp = data_min_all.groupby(['stationID','week','hour','minute'], as_index=False)['inNums'].agg({
                                                                        'inNums_whm_max'    : 'max',
                                                                        'inNums_whm_min'    : 'min',
                                                                        'inNums_whm_mean'   : 'mean',
                                                                        'inNums_whm_std'   : 'std'
                                                                        })
data_min_all = data_min_all.merge(tmp, on=['stationID','week','hour','minute'], how='left')

tmp = data_min_all.groupby(['stationID','week','hour','minute'], as_index=False)['outNums'].agg({
                                                                        'outNums_whm_max'    : 'max',
                                                                        'outNums_whm_min'    : 'min',
                                                                        'outNums_whm_mean'   : 'mean',
                                                                        'outNums_whm_std'   : 'std'
                                                                        })
data_min_all = data_min_all.merge(tmp, on=['stationID','week','hour','minute'], how='left')

tmp = data_min_all.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_wh_max'    : 'max',
                                                                        'inNums_wh_min'    : 'min',
                                                                        'inNums_wh_mean'   : 'mean',
                                                                        'inNums_wh_std'   : 'std'
                                                                        })
data_min_all = data_min_all.merge(tmp, on=['stationID','week','hour'], how='left')

tmp = data_min_all.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                        'outNums_wh_max'    : 'max',
                                                                        'outNums_wh_min'    : 'min',
                                                                        'outNums_wh_mean'   : 'mean',
                                                                        'outNums_wh_std'   : 'std'
                                                                        })
data_min_all = data_min_all.merge(tmp, on=['stationID','week','hour'], how='left')
data_min_all = data_min_all.fillna(0)

#地铁运营外部数据
station_detail = pd.read_excel("F:/代码空间/1903地铁预测/station_id.xlsx")
data_min_all_temp = data_min_all.merge(station_detail,on="stationID",how="left")
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
data_min_all = data_min_all_temp
#去除前5天数据
data_min_all = data_min_all[(data_min_all.day != 2) & (data_min_all.day != 3) & (data_min_all.day != 4)
                & (data_min_all.day != 7)& (data_min_all.day != 8)]
all_columns = [f for f in data_min_all.columns if f not in ['weekend','inNums','outNums'
    , "nuni_deviceID_of_stationID", "nuni_deviceID_of_stationID_hour"]]

### all data
all_data = data_min_all[data_min_all.day!=29]
X_data = all_data[all_columns].values

train = data_min_all[data_min_all.day <28]
X_train = train[all_columns].values

valid = data_min_all[data_min_all.day==28]
X_valid = valid[all_columns].values

test  = data_min_all[data_min_all.day==29]
X_test = test[all_columns].values

#构建模型并训练
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'max_depth':5,
    'num_leaves': 30,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    # "min_data_in_leaf": 1,
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
                num_boost_round=8000,
                valid_sets=[lgb_train,lgb_evals],
                valid_names=['train','valid'],
                early_stopping_rounds=50,
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
                num_boost_round=8000,
                valid_sets=[lgb_train,lgb_evals],
                valid_names=['train','valid'],
                early_stopping_rounds=50,
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

sub29=pd.read_csv(open('F:/数据集/1903地铁预测/Metro_testA/testA_submit_2019-01-29.csv',encoding='utf8'))
del sub29['inNums']
del sub29['outNums']
sub29['hour']=pd.to_datetime(sub29['startTime'],format='%Y-%m-%d %H:%M:%S').dt.hour
sub29['minute']=pd.to_datetime(sub29['startTime'],format='%Y-%m-%d %H:%M:%S').dt.minute
sub29=sub29.merge(test[["stationID","hour","minute","inNums","outNums"]],on = ['stationID', 'hour', 'minute'],how = 'left')
submition=sub29[['stationID','startTime','endTime','inNums','outNums']]
submition.loc[submition["stationID"]==54,"inNums"]=0
submition.loc[submition["stationID"]==54,"outNums"]=0
submition.to_csv('F:/数据集处理/1903地铁预测/submit/lgb_min_pre29_0328.csv',index=False)
