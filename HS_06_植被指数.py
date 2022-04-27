# -*- coding: utf-8 -*-
r'''
Created on Wed Nov  7 18:35:04 2019
@author: yangjing
-----------------------------------------------------------------------------
该脚本用于获取植被指数：

[注意]
          -svc SVC的excel文件
          -spad与svc必须一一对应，数量相等

'''
print(__doc__)
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
import os
import argparse
import xlrd
import time

starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s' % time.ctime())
print('-' * 80)
parse = argparse.ArgumentParser(description='add svc file and spad file')
parse.add_argument('-svc', '--svc', help='the SVC  FILE')

args = parse.parse_args()

full_path = args.svc

HS_path = os.path.split(full_path)[0]

print('若长度不相等，按 Ctrl+C 终止')


# 1：归一化植被指数S
def NDVI(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][670]) / (data.iloc[i][800] + data.iloc[i][670]) for i in range(LENGTH)]
    return l
# 2：土壤调和植被指数SAVI
def I_1(data, LENGTH):
    RC = 0.5
    l = [(1 + RC) * (data.iloc[i][800] - data.iloc[i][670]) / (data.iloc[i][800] + data.iloc[i][670] + RC) for i in
         range(LENGTH)]
    return l


# 6：GRVI
def I_2(data, LENGTH):
    l = [(data.iloc[i][620] - data.iloc[i][506]) / (data.iloc[i][620] + data.iloc[i][506]) for i in range(LENGTH)]
    return l


# 8: PRI
def I_3(data, LENGTH):
    l = [(data.iloc[i][570] - data.iloc[i][532]) / (data.iloc[i][570] + data.iloc[i][532]) for i in range(LENGTH)]
    return l


# 9: NPCI
def I_4(data, LENGTH):
    l = [(data.iloc[i][642] - data.iloc[i][432]) / (data.iloc[i][642] + data.iloc[i][432]) for i in range(LENGTH)]
    return l


# 11: mSR
def I_5(data, LENGTH):
    l = [(data.iloc[i][750] - data.iloc[i][446]) / (data.iloc[i][706] + data.iloc[i][446]) for i in range(LENGTH)]
    return l


# 12: PPR
def I_6(data, LENGTH):
    l = [(data.iloc[i][504] - data.iloc[i][436]) / (data.iloc[i][504] + data.iloc[i][436]) for i in range(LENGTH)]
    return l


# 12: SIPI
def I_7(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][446]) / (data.iloc[i][800] - data.iloc[i][680]) for i in range(LENGTH)]
    return l


# 13: NDWI
def I_8(data, LENGTH):
    l = [(data.iloc[i][858] - data.iloc[i][1240]) / (data.iloc[i][858] + data.iloc[i][1240]) for i in range(LENGTH)]
    return l


### NDSI
def I_9(data, LENGTH):
    l = [(data.iloc[i][814] - data.iloc[i][762]) / (data.iloc[i][814] + data.iloc[i][762]) for i in range(LENGTH)]
    return l


# LCI
def I_10(data, LENGTH):
    l = [(data.iloc[i][850] - data.iloc[i][710]) / (data.iloc[i][850] - data.iloc[i][680]) for i in range(LENGTH)]
    return l
# RVI
def I_11(data, LENGTH):
    l = [(data.iloc[i][766]) / (data.iloc[i][720]) for i in range(LENGTH)]
    return l

# 以上部分写入植被指数的计算式

def corr():
    print('[INFO]正在处理，请稍后...')
    workbook = xlrd.open_workbook(full_path)
    names = workbook.sheet_names()

    for period in names:
        print('正在处理第 %s 期' % period)
        # water=pd.read_excel(water_path,sheet_name=period,header=None)
        data = pd.read_excel(full_path, sheet_name=period)
        LENGTH = len(data.index)
        d = {'NDVI': NDVI(data, LENGTH), 'RVI': I_11(data, LENGTH), 'GRVI': I_2(data, LENGTH), 'PRI': I_3(data, LENGTH),
            'NPCI': I_4(data, LENGTH), 'mSR': I_5(data, LENGTH),
             'PPR': I_6(data, LENGTH),
             'SIPI': I_7(data, LENGTH),'NDSI': I_9(data, LENGTH), 'LCI': I_10(data, LENGTH)}
# 这里按照上面的格式写入输出的植被指数名字
        index = data.index
        INDEX = pd.DataFrame(d, index=index)
        col = INDEX.columns

        # INDEX_1=pd.concat([water,INDEX],axis=1)
        INDEX.to_excel(os.path.join(HS_path, '%s植被指数.xlsx' % period))


def main():
    print('[INFO]加载数据集...')
    corr()
    endtime = datetime.datetime.now()
    print('-' * 60)
    print('程序运行时间:%s s' % ((endtime - starttime).seconds))


main()
