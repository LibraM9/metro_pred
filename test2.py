# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : test2.py
# @time  : 2019/3/27
"""
文件说明：
"""
import lightgbm as lgb
import pandas as pd
import os
import tqdm
import numpy as np
from scipy.stats import skew
from tsfresh.feature_extraction import feature_calculators as ts
from tsfresh.feature_extraction import extract_features

def _get_sts_features_s(data):
    #### 1.object_id:count ####

    grp_col = 'district_code'

    city_code = 'city_code'

    df = pd.DataFrame()

    df[grp_col] = data[grp_col].values

    df[city_code] = data[city_code].values

    df['ranks'] = data['ranks'].values

    ### 过去N天的统计特征 ####

    for fea_col in tqdm(['flow_in', 'flow_out', 'dwell']):

        for slide_windows in [3, 6, 13, 20, 27]:

            print(fea_col, slide_windows)

            slide_cols = [fea_col + '_before_' + str(i + 1) for i in range(slide_windows)]

            slide_cols.append(fea_col)

            df_tmp = data[slide_cols].values

            df_tmp_percent = data[slide_cols].copy()

            df_tmp_percent['sum_'] = df_tmp_percent.sum(axis=1).values

            for col in slide_cols:
                df_tmp_percent[col] = df_tmp_percent[col].values / (1e-5 + df_tmp_percent['sum_'].values)  # 百分比

            df_tmp_percent = df_tmp_percent[slide_cols].values

            df_grp = data.groupby(city_code)[slide_cols].sum(axis=1).reset_index()

            df_city_dic = data.groupby(city_code)[slide_cols].sum(axis=1).to_dict()

            df['district_' + fea_col + '_last{}_sum'.format(slide_windows)] = np.sum(df_tmp, axis=1)

            df['district_' + fea_col + '_last{}_median'.format(slide_windows)] = np.median(df_tmp, axis=1)

            df['district_' + fea_col + '_last{}_std'.format(slide_windows)] = np.std(df_tmp, axis=1)

            df['district_' + fea_col + '_last{}_min'.format(slide_windows)] = np.min(df_tmp, axis=1)

            df['district_' + fea_col + '_last{}_max'.format(slide_windows)] = np.max(df_tmp, axis=1)

            df['district_' + fea_col + '_last{}_mean_change'.format(slide_windows)] = data[slide_cols].apply(
                ts.mean_change, axis=1)

            df['district_percent_' + fea_col + '_last{}_median'.format(slide_windows)] = np.median(df_tmp_percent,
                                                                                                   axis=1)

            df['district_percent_' + fea_col + '_last{}_std'.format(slide_windows)] = np.std(df_tmp_percent, axis=1)

            df['district_percent_' + fea_col + '_last{}_min'.format(slide_windows)] = np.min(df_tmp_percent, axis=1)

            df['district_percent_' + fea_col + '_last{}_max'.format(slide_windows)] = np.max(df_tmp_percent, axis=1)

            df['district_percent_' + fea_col + '_last{}_skew'.format(slide_windows)] = skew(df_tmp, axis=1)

    df['flow_in_last_week'] = data['flow_in_before_7'].values

    df['flow_out_last_week'] = data['flow_out_before_7'].values

    df['dwell_last_week'] = data['dwell_before_7'].values

    for fea_col in tqdm(['flow_in', 'flow_out', 'dwell']):

        for slide_windows in range(1, 4):

            print(fea_col, slide_windows)

            slide_cols = [fea_col + '_before_' + str(i * 7 + 7) for i in range(slide_windows)]

            slide_cols.append(fea_col)

            df_tmp = data[slide_cols].values

            if slide_windows == 1:

                df[fea_col + '_w{}_mean'.format(slide_windows)] = np.mean(df_tmp, axis=1)

            else:

                df[fea_col + '_w{}_mean'.format(slide_windows)] = np.mean(df_tmp, axis=1)

                df[fea_col + '_w{}_median'.format(slide_windows)] = np.median(df_tmp, axis=1)

                df[fea_col + '_w{}_std'.format(slide_windows)] = np.std(df_tmp, axis=1)

                df[fea_col + '_w{}_mean_change'.format(slide_windows)] = data[slide_cols].apply(ts.mean_change, axis=1)

    return df
