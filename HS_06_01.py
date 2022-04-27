r'''
	该脚本用于合并：【S_derivative.xlsx,Z_derivative.xlsx,X_derivative.xlsx】
	合并：【S_three_edge.xlsx,Z_three_edge.xlsx,X_three_edge.xlsx】
'''
import pandas as pd
import datetime
import os
import argparse
import time
print(__doc__)

starttime = datetime.datetime.now()

parse = argparse.ArgumentParser ( description='add svc file and spad file' )
parse.add_argument ( '-svc', '--svc', help='the SVC  FILE' )

args = parse.parse_args ()
full_path = args.svc

try:
	os.makedirs(os.path.join(full_path,'HS_06_01'))
except:
	pass
HS_path=os.path.join(full_path,'HS_06_01')

##针对玉米上中下分层数据，对S_derivative.xlsx, Z_derivative.xlsx , X_derivative.xlsx进行合并
##注意根据需要设置传入参数

print('Current time:%s'%time.ctime())
print('-'*80)

def combine():
	files=['S_derivative.xlsx','Z_derivative.xlsx' ,'X_derivative.xlsx']
	three=['S_three_edge.xlsx','Z_three_edge.xlsx','X_three_edge.xlsx']
	S=pd.read_excel(full_path+'/'+files[0])
	Z=pd.read_excel(full_path+'/'+files[1])
	X=pd.read_excel(full_path+'/'+files[2])
	length=len(S)+len(Z)+len(X)

	S_three=pd.read_excel(full_path+'/'+three[0])
	Z_three=pd.read_excel(full_path+'/'+three[1])
	X_three=pd.read_excel(full_path+'/'+three[2])

	data1=pd.concat([S,Z,X])
	data1=data1[data1.columns[1:]]
	data1.index=range(length)

	data2=pd.concat([S_three,Z_three,X_three])
	data2=data2[data2.columns[1:]]
	data2.index=range(length)

	data1.to_excel(os.path.join(HS_path,'derivative.xlsx'))
	data2.to_excel(os.path.join(HS_path, 'three_edge.xlsx'))

combine()
endtime = datetime.datetime.now()
print('-' * 60)
print('程序运行时间:%s s' % ((endtime - starttime).seconds))