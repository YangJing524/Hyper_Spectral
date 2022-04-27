r'''
	HS_08.py
	由于误差的普遍存在，
	我们拟采用通用的办法--四分位法对原始数据的偏离值做一个预处理
	对于高于上四分位、以及低于下四分位的SPAD以及SVC数据同步剔除后，
	重新计算相关性
'''

print(__doc__)
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
import os
import argparse
import time
starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s'%time.ctime())
print('-'*80)
parse = argparse.ArgumentParser ( description='add svc file and spad file' )
parse.add_argument ( '-svc', '--svc', help='the SVC  FILE' )
parse.add_argument ( '-spad', '--spad', help='the SPAD EXCEL FILE' )


args = parse.parse_args ()
svc_path = args.svc
spad_path=args.spad

##此处手动指定需要删除的列表
delete_list=[59, 60, 121, 123, 126, 175, 184, 191, 207, 210, 227, 364, 421, 463, 519]


data=pd.read_excel(svc_path, index_col=None)
spad=pd.read_excel(spad_path, header=None)

data1=data.copy()
spad1=spad.copy()

data1=data1.drop(delete_list)
spad1=spad1.drop(delete_list)

LENGTH=len(data.index)
print('原始 svc 长度:%s' % len(data))
print('原始 spad 长度:%s' % len(spad))
print('剔除异常值的 svc 长度: %s' % len(data1))
print('剔除异常值的 spad 长度: %s' % len(spad1))
try:
	os.mkdir(os.path.join(os.path.split(svc_path)[0], 'HS_08'))
except:
	pass

HS_path=os.path.join(os.path.split(svc_path)[0], 'HS_08')


def corr():
	print('[INFO]正在处理，请稍后...')
	num=len(spad1.index)
	index=np.linspace(0,num-1,num)

	col=data1.columns
	spad1.index=index
	spad1.name='SPAD'
	for i in col:
		#pearsonr函数返回两个值，分别为相关系数以及P值（显著性水平）
		#l1:相关系数列表，l2:p值列表
		value=pearsonr(spad1[spad1.columns[0]], data1[i])
		with open(os.path.join(HS_path, 'INDEX.txt'), 'a+') as f:
			f.writelines('参数：%s，相关系数：%s,p水平检验：%s'%(i, value[0], value[1])+'\n')


def main():

	print('[INFO]加载数据集...')
	corr()
	endtime = datetime.datetime.now()
	print('-'*60)
	print('程序运行时间:%s s'%( ( endtime - starttime ). seconds ))


main()
