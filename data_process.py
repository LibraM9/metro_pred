import pandas as pd
import datetime
import math
from datetime import timedelta


input_path = 'F:/数据集/1903地铁预测/Metro_train/'
output = 'F:/数据集处理/1903地铁预测/train/'

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

def trans(in_file_name):
    out_file_name = in_file_name.split('_')[0]+'_'+in_file_name.split('_')[-1]

    f = open(input_path+in_file_name, encoding='utf-8')
    record_01 = pd.read_csv(f)

    record_01['startTime'] = record_01.apply(startTime, axis=1)
    record_01['endTime'] = record_01.apply(endTime, axis=1)

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

    # df.to_csv(output+out_file_name)
    return df

for i in range(1,26):
    in_file_name = 'record_2019-01-{}.csv'.format("0" + str(i) if len(str(i)) == 1 else str(i))
    print(in_file_name)
    if i == 1:
        df = trans(in_file_name)
    else:
        df = pd.concat([df,trans(in_file_name)],axis=0)

ans = df.reset_index(drop=True)
ans.to_csv(output+"train.csv")
