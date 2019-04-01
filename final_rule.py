# -*- coding: utf-8 -*-
#@author: limeng
#@file: final_rule.py
#@time: 2019/3/30 20:33
"""
文件说明：淘汰赛规则
"""
import pandas as pd
import numpy as np
train = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all_b.csv",encoding="utf8"))
#周末分钟均值
holiday = [6,13,20]
t_holiday=train[train['day'].isin(holiday)]
t_holiday_in=t_holiday.groupby(['stationID', 'hour', 'minute'])['inNums'].agg({'in_mean':'mean'}).reset_index()
t_holiday_out=t_holiday.groupby(['stationID', 'hour', 'minute'])['outNums'].agg({'out_mean':'mean'}).reset_index()
t_holiday_in_out=t_holiday_in.merge(t_holiday_out,on = ['stationID', 'hour', 'minute'],how = 'left')
t_holiday_in_out.dtypes

#周末小时均值
t_holiday_in_hour=t_holiday.groupby(['stationID', 'hour'])['inNums'].agg({'in_mean_hour':'mean'}).reset_index()
t_holiday_out_hour=t_holiday.groupby(['stationID', 'hour'])['outNums'].agg({'out_mean_hour':'mean'}).reset_index()
t_holiday_in_out_hour=t_holiday_in_hour.merge(t_holiday_out_hour,on = ['stationID', 'hour'],how = 'left')
t_holiday_in_out_hour.dtypes

#分钟平均占小时的比例 hour*n=minute
t_holiday_in_out=t_holiday_in_out.merge(t_holiday_in_out_hour,on = ['stationID', 'hour'],how = 'left')
t_holiday_in_out['in_bili']=t_holiday_in_out['in_mean']/(t_holiday_in_out['in_mean_hour']+0.001)
t_holiday_in_out['out_bili']=t_holiday_in_out['out_mean']/(t_holiday_in_out['out_mean_hour']+0.001)
t_holiday_in_out_bili=t_holiday_in_out[['stationID', 'hour', 'minute','in_bili','out_bili','in_mean_hour','out_mean_hour']]

#周日对于周六的小时比例,6*n=7
feature = ['stationID', 'hour', 'minute',"inNums","outNums"]
d_12 = train.loc[train['day']==12,feature].rename(columns={"inNums":"inNums_12","outNums":"outNums_12"})
d_13 = train.loc[train['day']==13,feature].rename(columns={"inNums":"inNums_13","outNums":"outNums_13"})
d_19 = train.loc[train['day']==19,feature].rename(columns={"inNums":"inNums_19","outNums":"outNums_19"})
d_20 = train.loc[train['day']==20,feature].rename(columns={"inNums":"inNums_20","outNums":"outNums_20"})
t_67_bili =d_12.merge(d_13,on=['stationID', 'hour', 'minute'],how="left")
t_67_bili =t_67_bili.merge(d_19,on=['stationID', 'hour', 'minute'],how="left")
t_67_bili =t_67_bili.merge(d_20,on=['stationID', 'hour', 'minute'],how="left")
t_67_bili=t_67_bili.groupby(['stationID', 'hour'])[
    "inNums_12","outNums_12","inNums_13","outNums_13"
,"inNums_19","outNums_19","inNums_20","outNums_20"].agg(["mean"]).reset_index()
t_67_bili.columns = [i[0]+i[1] for i in t_67_bili.columns]
t_67_bili["in_bili_67"]= ((t_67_bili["inNums_20mean"]/(t_67_bili["inNums_19mean"]))\
                         +(t_67_bili["inNums_13mean"]/(t_67_bili["inNums_12mean"])))/2
t_67_bili["out_bili_67"]= ((t_67_bili["outNums_20mean"]/(t_67_bili["outNums_19mean"]))\
                         +(t_67_bili["outNums_13mean"]/(t_67_bili["outNums_12mean"])))/2
t_67_bili = t_67_bili.fillna(1)
t_67_bili = t_67_bili.replace(np.inf,1)
ans = t_67_bili[t_67_bili.hour.isin([6,7,8,9,10,11,12,13,14,1,51,61,71,8,19,20,21,22,23])]

#第26天 小时均值
t26=train[(train['day'] == 26)]

t26_in_hour=t26.groupby(['stationID', 'hour'])['inNums'].agg({'in26_mean_hour': 'mean'}).reset_index()
t26_out_hour=t26.groupby(['stationID', 'hour'])['outNums'].agg({'out26_mean_hour': 'mean'}).reset_index()
t26_in_out_hour=t26_in_hour.merge(t26_out_hour,on = ['stationID', 'hour'],how = 'left')

t26=t26.merge(t26_in_out_hour, on = ['stationID', 'hour'], how ='left')
t_26=t26[['stationID', 'hour', 'minute', 'inNums', 'outNums', 'in26_mean_hour', 'out26_mean_hour']]

#第27天
sub27=pd.read_csv(open('F:/数据集/1903地铁预测/Metro_testB/testB_submit_2019-01-27.csv', encoding="utf8"))
del sub27['inNums']
del sub27['outNums']
sub27.dtypes

sub27['hour']=pd.to_datetime(sub27['startTime'], format='%Y-%m-%d %H:%M:%S').dt.hour
sub27['minute']=pd.to_datetime(sub27['startTime'], format='%Y-%m-%d %H:%M:%S').dt.minute

sub27=sub27.merge(t_holiday_in_out_bili, on = ['stationID', 'hour', 'minute'], how ='left')
sub27=sub27.merge(t_26, on = ['stationID', 'hour', 'minute'], how ='left')
sub27=sub27.merge(t_67_bili, on = ['stationID', 'hour'], how ='left')
sub27=sub27.fillna(0)
"""
in26_mean_hour out26_mean_hour 26日小时均值
in_bili_67  out_bili_67 周日占周六比例
in_bili out_bili 分钟占小时比例
"""
sub27['in27_26mean']=sub27['in26_mean_hour']*sub27['in_bili_67']*sub27['in_bili'] #每分钟占每小时比例*小时平均值
sub27['out27_26mean']=sub27['out26_mean_hour']*sub27['out_bili_67']*sub27['out_bili']
submition=sub27[['stationID','startTime','endTime','in27_26mean','out27_26mean']]
submition=submition.rename(columns = {'in27_26mean':'inNums'})
submition=submition.rename(columns = {'out27_26mean':'outNums'})
submition.to_csv('F:/数据集处理/1903地铁预测/submit/final_sub/sub1_rule_bili.csv',index=False)#特殊站使用规则，非特殊站使用模型


#####stacking
# model = pd.read_csv(open('F:/数据集处理/1903地铁预测/submit/final_sub/sub2_lgb_67.csv'))
model = pd.read_csv(open('F:/数据集处理/1903地铁预测/submit/final_sub/new_lgb_testB_sub.csv'))
model2 = pd.read_csv(open('F:/数据集处理/1903地铁预测/submit/final_sub/sub2_lgb_67.csv'))
stack = model.merge(model2,on=["stationID","startTime","endTime"],how="left")
stack["inNums"] = stack["inNums_x"]*0.6+stack["inNums_y"]*0.4
stack["outNums"] = stack["outNums_x"]*0.6+stack["outNums_y"]*0.4
# rule1 = pd.read_csv(open('F:/数据集处理/1903地铁预测/submit/final_sub/sub1_rule_bili.csv'))
# rule2 = pd.read_csv(open('F:/数据集处理/1903地铁预测/submit/final_sub/submit27.csv'))
# stack = rule1.merge(rule2,on=["stationID","startTime","endTime"],how="left")
# stack = stack.merge(model,on=["stationID","startTime","endTime"],how="left")
# stack["inNums"] = stack["inNums_x"]*0.25+stack["inNums_y"]*0.25+stack["inNums"]*0.5
# stack["outNums"] = stack["outNums_x"]*0.25+stack["outNums_y"]*0.25+stack["outNums"]*0.5
stack[['stationID','startTime','endTime','inNums','outNums']].to_csv("F:/数据集处理/1903地铁预测/submit/final_sub/sub7_lgb67_1227_stack64.csv",index=False)

#modify
stack = pd.read_csv(open('F:/数据集处理/1903地铁预测/submit/final_sub/sub3_rule_lgb_stack.csv'))
stack['hour']=pd.to_datetime(stack['startTime'], format='%Y-%m-%d %H:%M:%S').dt.hour
stack['minute']=pd.to_datetime(stack['startTime'], format='%Y-%m-%d %H:%M:%S').dt.minute
train = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/data_all_b.csv",encoding="utf8"))
d_20 = train.loc[train['day']==20,feature].rename(columns={"inNums":"inNums_20","outNums":"outNums_20"})
stack = stack.merge(d_20,how="left",on=["stationID","hour","minute"])
def modify(df):
    df["inNums"] = df["inNums_20"]
    df["outNums"] = df["outNums_20"]
    return df
stack.loc[stack["hour"].isin([0,1,2,3,4]),:]=stack.loc[stack["hour"].isin([0,1,2,3,4]),:].apply(modify,axis=1)
stack[['stationID','startTime','endTime','inNums','outNums']].to_csv("F:/数据集处理/1903地铁预测/submit/final_sub/sub3_rule_lgb_stack.csv",index=False)
