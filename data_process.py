import pandas as pd
import numpy as np
import datetime
import math
from datetime import timedelta


input_path = 'E:/Competition/天池/Metro/data/Metro_train/'
output = 'E:/Competition/天池/Metro/data/Metro_train_record/'

in_file_name = 'record_2019-01-01.csv'
out_file_name = in_file_name.split('_')[0]+'_'+in_file_name.split('_')[-1]

f = open(input_path+in_file_name, encoding='utf-8')
record_01 = pd.read_csv(f)


def startTime(df):

    pass_time = datetime.datetime.strptime(df['time'], "%Y-%m-%d %H:%M:%S")
    minute = int(math.floor(pass_time.minute / 10) * 10)
    start_time = datetime.datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, minute, 0)
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    return start_time


def endTime(df):
    start_time = datetime.datetime.strptime(df['startTime'], "%Y-%m-%d %H:%M:%S")
    end_time = start_time + timedelta(minutes=10)
    return end_time



record_01['startTime'] = record_01.apply(startTime, axis=1)
record_01['endTime'] = record_01.apply(endTime, axis=1)

# record_01['time'] = pd.to_datetime(record_01['time'])
# record_01['weekend'] = (record_01['time'].dt.weekday >= 5).astype(int)
# record_01['dayofweek'] = record_01['time'].dt.dayofweek
# record_01['day'] = record_01['time'].dt.day
# record_01['hour'] = record_01['time'].dt.hour


# time,lineID,stationID,deviceID,status,userID,payType
agg_fun = {
    'lineID':['nunique','count'],
    'deviceID':['nunique', 'count'],
    'payType':['nunique']
}

df = record_01.groupby(['stationID', 'startTime', 'endTime']).agg(agg_fun).\
    sort_values(by=['stationID', 'startTime', 'endTime'], ascending=True)
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.reset_index(inplace=True)


status_1 = record_01[record_01['status']==1]
status_1 = status_1[['stationID', 'startTime', 'endTime','userID']]
status_0 = record_01[record_01['status'] == 0]
status_0 = status_0[['stationID', 'startTime', 'endTime','userID']]

df['inNums'] = status_1.groupby(['stationID', 'startTime', 'endTime'], as_index=False).agg({'userID':'count'}).sort_values(by=['stationID', 'startTime', 'endTime'], ascending=True)['userID']
df['outNums'] = status_0.groupby(['stationID', 'startTime', 'endTime'], as_index=False).agg({'userID':'count'}).sort_values(by=['stationID', 'startTime', 'endTime'], ascending=True)['userID']

df.to_csv(output+out_file_name)
