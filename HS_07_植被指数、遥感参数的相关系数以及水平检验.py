# -*- coding: utf-8 -*-
r'''
Created on Wed Nov  7 18:35:04 2019
@author: yangjing
数据整理样式：
第1列为SPAD 或 叶片含水量, 且标题必须为 SPAD 

	SPAD	Dr	Db	Dy	SDr	SDb	SDy		SDr/SDb
0	41.1	718	529	579	0.4284	0.0952	-0.0936		4.5
1	26.3	714	529	582	0.4426	0.1198	-0.11965	3.694490818
2	36.8	714	530	579	0.4278	0.0942	-0.09975	4.541401274
3	33.6	714	530	582	0.39515	0.1002	-0.10425	3.943612774
4	28.2	719	530	576	0.37945	0.07325	-0.0753		5.180204778
5	27.8	714	529	578	0.4097	0.10195	-0.1017		4.018636587
6	39.4	714	530	576	0.44235	0.10185	-0.1041		4.343151694
7	33	719	530	576	0.43485	0.07825	-0.08135	5.557188498
8	34.8	714	530	578	0.41245	0.1056	-0.1074		3.905776515
9	31.8	718	529	577	0.43925	0.0893	-0.09215	4.91881299
10	38.1	714	528	579	0.4044	0.12325	-0.1199		3.281135903

-----------------------------------------------------------------------------
该脚本用于获取三边参数或水平检验与 SPAD 之间的相关性与水平检验，结果写入到corr_水平检验.csv

[注意]

          -spad与svc必须一一对应，数量相等

'''
print(__doc__)
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
import os
import argparse
import xlrd
import time
import csv
starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s'%time.ctime())
print('-'*80)
'''
parse = argparse.ArgumentParser ( description='add svc file and spad file' )
parse.add_argument ( '-svc', '--svc', help='the SVC  FILE' )
parse.add_argument ( '-spad', '--spad', help='the SPAD EXCEL FILE' )


args = parse.parse_args ()
svc_path = args.svc
spad_path=args.spad
'''

path=r'J:\论文数据\output\water\water.xlsx'

HS_path=os.path.split(path)[0]


def corr(spad,data,period):
	print('[INFO]正在处理，请稍后...')

	l=[]
	d={}
	col=data.columns[1:]
	for i in col:
		#pearsonr函数返回两个值，分别为相关系数以及P值（显著性水平）
		#l1:相关系数列表，l2:p值列表
		value=pearsonr(spad, data[i])
		if value[1]<0.01:
			d[i]='%s**'%round(value[0],3)
		elif value[1]<0.05:
			d[i]='%s*'%round(value[0],3)
		else:
			d[i]=round(value[0],3)
	l.append(d)
	
	headers=list(col)
	with open(os.path.join(HS_path, '%s_corr_水平检验.csv'%period), 'w', newline='') as f:
		writer=csv.DictWriter(f,headers)
		writer.writeheader()
		writer.writerows(l)


def main():
	print('[INFO] 加载数据集...')
	
	periods=['0605','0622','0717','0826']
	for key in periods:
		print('正在处理第 %s 期'%key)
		data=pd.read_excel(path,sheet_name=key)
		water=data['water']
		corr(water,data,key)
		
	endtime = datetime.datetime.now()
	print('-'*60)
	print('程序运行时间:%s s'%( ( endtime - starttime ). seconds ))


main()
