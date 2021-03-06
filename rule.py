# -*- coding: utf-8 -*-
#@author: limeng
#@file: rule.py
#@time: 2019/3/23 22:00
"""
文件说明：规则
"""
import pandas as pd
#train0就是1-25号所有统计的记录，time就是starttime，按照要求作出十分钟级别的。
ins= pd.read_csv(open('F:/数据集处理/1903地铁预测/train/in.csv',encoding='utf8'))
outs = pd.read_csv(open('F:/数据集处理/1903地铁预测/train/out.csv',encoding='utf8'))

train0=ins.merge(outs,on = ['stationID', 'time'],how = 'inner')

train0['days']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.day
train0['hours']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.hour
train0['minutes']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.minute
train0['wkday']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.weekday
train0['wk']=train0['wkday']+1

# train0['days']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.day
# train0['hours']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.hour
# train0['minutes']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.minute
# train0['wkday']=pd.to_datetime(train0['time'],format='%Y-%m-%d %H:%M:%S').dt.weekday

#前一周 周一到周五的分钟均值
train = train0.copy()
t21_25=train[(train['days']>=21)&(train['days']<=25)]
t21_25_in=t21_25.groupby(['stationID', 'hours', 'minutes'])['inNums'].agg({'in_mean':'mean'}).reset_index()
t21_25_out=t21_25.groupby(['stationID', 'hours', 'minutes'])['outNums'].agg({'out_mean':'mean'}).reset_index()
t21_25_in_out=t21_25_in.merge(t21_25_out,on = ['stationID', 'hours', 'minutes'],how = 'left')
t21_25_in_out.dtypes

#前一周 周一到周五小时均值
t21_25_in_hour=t21_25.groupby(['stationID', 'hours'])['inNums'].agg({'in_mean_hour':'mean'}).reset_index()
t21_25_out_hour=t21_25.groupby(['stationID', 'hours'])['outNums'].agg({'out_mean_hour':'mean'}).reset_index()
t21_25_in_out_hour=t21_25_in_hour.merge(t21_25_out_hour,on = ['stationID', 'hours'],how = 'left')
t21_25_in_out_hour.dtypes

#分钟平均占小时的比例
t21_25_in_out=t21_25_in_out.merge(t21_25_in_out_hour,on = ['stationID', 'hours'],how = 'left')
t21_25_in_out['in_bili']=t21_25_in_out['in_mean']/(t21_25_in_out['in_mean_hour']+0.001)
t21_25_in_out['out_bili']=t21_25_in_out['out_mean']/(t21_25_in_out['out_mean_hour']+0.001)
t21_25_in_out_bili=t21_25_in_out[['stationID', 'hours', 'minutes','in_bili','out_bili','in_mean_hour','out_mean_hour']]

#第28天 小时均值
t28=train[(train['days']==28)]

t28_in_hour=t28.groupby(['stationID', 'hours'])['inNums'].agg({'in28_mean_hour':'mean'}).reset_index()
t28_out_hour=t28.groupby(['stationID', 'hours'])['outNums'].agg({'out28_mean_hour':'mean'}).reset_index()
t28_in_out_hour=t28_in_hour.merge(t28_out_hour,on = ['stationID', 'hours'],how = 'left')

t28=t28.merge(t28_in_out_hour,on = ['stationID', 'hours'],how = 'left')
t_28=t28[['stationID', 'hours', 'minutes','inNums','outNums','in28_mean_hour','out28_mean_hour']]

#第29天
sub29=pd.read_csv(open('F:/数据集/1903地铁预测/Metro_testA/testA_submit_2019-01-29.csv',encoding='utf8'))
del sub29['inNums']
del sub29['outNums']
sub29.dtypes

sub29['hours']=pd.to_datetime(sub29['startTime'],format='%Y-%m-%d %H:%M:%S').dt.hour
sub29['minutes']=pd.to_datetime(sub29['startTime'],format='%Y-%m-%d %H:%M:%S').dt.minute

sub29=sub29.merge(t21_25_in_out_bili,on = ['stationID', 'hours', 'minutes'],how = 'left')
sub29=sub29.merge(t_28,on = ['stationID', 'hours', 'minutes'],how = 'left')
sub29=sub29.fillna(0)

#lgb预测的每小时出入人数
# lgb_hour_pre29 = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/lgb_hour_pre29_0328.csv",encoding="utf8"))
# lgb_hour_pre29.columns=["stationID","hours","in_hour_pre_lgb","out_hour_pre_lgb"]
# sub29 = sub29.merge(lgb_hour_pre29,on=["stationID","hours"],how="left")
# sub29["in_hour_pre_lgb"] = sub29["in_hour_pre_lgb"]/6
# sub29["out_hour_pre_lgb"] = sub29["out_hour_pre_lgb"]/6
#
# #fbprophet预测的每小时人数
# fb_hour_pre29 = pd.read_csv(open("F:/数据集处理/1903地铁预测/train/fb_hour_pre29.csv",encoding="utf8"))
# fb_hour_pre29.columns=["stationID","hours","in_hour_pre_fb","out_hour_pre_fb"]
# sub29 = sub29.merge(fb_hour_pre29,on=["stationID","hours"],how="left")
# sub29["in_hour_pre_fb"] = sub29["in_hour_pre_fb"]/6
# sub29["out_hour_pre_fb"] = sub29["out_hour_pre_fb"]/6

sub29['in29_28mean']=sub29['in_bili']*sub29['in28_mean_hour']*0.965 #每分钟占每小时比例*小时平均值
sub29['out29_28mean']=sub29['out_bili']*sub29['out28_mean_hour']*0.965
# sub29['in29_fb']=sub29['in_bili']*sub29['in_hour_pre_fb'] #每分钟占每小时比例*小时平均值
# sub29['out29_fb']=sub29['out_bili']*sub29['out_hour_pre_fb']
# sub29['in29_lgb']=sub29['in_bili']*sub29['in_hour_pre_lgb'] #每分钟占每小时比例*小时平均值
# sub29['out29_lgb']=sub29['out_bili']*sub29['out_hour_pre_lgb']
# sub29.loc[sub29["stationID"]==7,'in29_lgb'] = sub29.loc[sub29["stationID"]==7,'in29_28mean'].values
# sub29.loc[sub29["stationID"]==7,'out29_lgb'] = sub29.loc[sub29["stationID"]==7,'out29_28mean'].values*1.1
# sub29.loc[sub29["stationID"]==15,'in29_lgb'] = sub29.loc[sub29["stationID"]==7,'in29_28mean'].values
# sub29.loc[sub29["stationID"]==15,'out29_lgb'] = sub29.loc[sub29["stationID"]==7,'out29_28mean'].values*1.05
#x=sub29[sub29['inNums'].isnull()]
#x=x[x['stationID']!=54]
submition=sub29[['stationID','startTime','endTime','in29_28mean','out29_28mean']]
submition=submition.rename(columns = {'in29_28mean':'inNums'})
submition=submition.rename(columns = {'out29_28mean':'outNums'})
submition.to_csv('F:/数据集处理/1903地铁预测/submit/sub_rule_bili_best.csv',index=False)#特殊站使用规则，非特殊站使用模型

#############stacking yu和bili
lgb_yu = pd.read_csv(open("F:/数据集处理/1903地铁预测/submit/lgb_min_yu.csv",encoding="utf8"))
rule_bili = pd.read_csv(open("F:/数据集处理/1903地铁预测/submit/sub_rule_bili_best.csv",encoding="utf8"))
submition = lgb_yu.merge(rule_bili,on=['stationID','startTime','endTime'],how="left")
submition["inNums"] = submition["inNums_x"]*0.5+submition["inNums_y"]*0.5
submition["outNums"] = submition["outNums_x"]*0.5+submition["outNums_y"]*0.5
submition=submition[['stationID','startTime','endTime','inNums','outNums']]
submition.to_csv('F:/数据集处理/1903地铁预测/submit/sub_rulebili_lgbmin_stack.csv',index=False)

##############stacking yu pro 和 bili
lgb_yu = pd.read_csv(open("F:/数据集处理/1903地铁预测/submit/lgb_min_yu_pro.csv",encoding="utf8"))
rule_bili = pd.read_csv(open("F:/数据集处理/1903地铁预测/submit/sub_rule_bili_best.csv",encoding="utf8"))
submition = lgb_yu.merge(rule_bili,on=['stationID','startTime','endTime'],how="left")
submition["inNums"] = submition["inNums_x"]*0.5+submition["inNums_y"]*0.5
submition["outNums"] = submition["outNums_x"]*0.5+submition["outNums_y"]*0.5
submition=submition[['stationID','startTime','endTime','inNums','outNums']]
submition.to_csv('F:/数据集处理/1903地铁预测/submit/sub_rulebili_lgbmin_stack_pro.csv',index=False)
