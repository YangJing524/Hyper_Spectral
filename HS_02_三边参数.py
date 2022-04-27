#coding=utf-8
u'''

USE:求取一阶导数、三边参数,分别输出derivate.xlsx与three_edge.xlsx
DOC:
	该脚本运行时间较长，请勿中途退出
'''


import pandas as pd
import time
import os
from tqdm import tqdm
import argparse
import xlrd
import math
import numpy as np

print(__doc__)
t=time.ctime()
print('Current time:%s'%time.ctime())
print('-'*80)


class Spectral:
	def __init__(self,path,name):
		self.path = path
		self.file_name = name

	def derivative(self):
		data = pd.read_excel(self.path,sheet_name=self.file_name)
		li=data.columns
		d = {}
		total = len(data)
		daoshu=1/data
		#kaifang=math.sqrt(data)
		duishu=np.log(data)
		#duishudedaoshu=1/np.log(data)
		#这里可以加自己想加的处理

		for i in range(total):
			l = []
			for j in range(li[0], li[-2],1):
				num = (data.iloc[i][j + 2] - data.iloc[i][j]) / 2
				l.append(num)
			d[i+1] = l

		first_data = pd.DataFrame(d)
		first_T = first_data.T
		output_path=os.path.split(self.path)[0]#os.path.dirname(path)
		# 这里写数据运算后的路径
		daoshu.to_excel(output_path + '/' + self.file_name + '_' + 'daoshu.xlsx')
		first_T.to_excel(output_path+'/'+self.file_name+'_'+'derivative.xlsx')
		duishu.to_excel(output_path + '/' + self.file_name + '_' + 'duishu.xlsx')

		#kaifang.to_excel(output_path + '/' + self.file_name + '_' + 'kaifang.xlsx')

		self.three_edge(first_T,output_path)

	def three_edge(self,first_T,output_path):
		d_three = {}

		total=len(first_T)
		with tqdm ( total=total ) as pbar:
			for i in range(total):
				data = first_T.iloc[i]
				l = []
				SDb=data[72:93].sum()
				l.append(SDb)#490-530nm    ([151, 191, 211, 241, 341，421]) (490-339=151:530-339=191)
				SDy=data[102:148].sum()
				l.append(SDy)#550-580nm----560-640
				SDr=data[167:208].sum()
				l.append(SDr)#
				#
				Db=(abs(data[72:93]).idxmax()-1)*2 + 350
				l.append(Db)
				Dy=(abs(data[102:148]).idxmax()-1)*2 + 350
				l.append(Dy)
				Dr=(abs(data[167:208]).idxmax()-1)*2 + 350
				l.append(Dr)

				l.append(data[72:93].max())
				l.append ( data[102:148].max () )
				l.append(data[167:208].max())

				l.append(SDr/SDb)
				l.append(SDr/SDy)
				l.append((SDr-SDb)/(SDr+SDb))
				l.append ( (SDr - SDy) / (SDr + SDy) )

				d_three[i] = l
				pbar.update(1)
		data2 = pd.DataFrame(d_three, index=['蓝边面积', '黄边面积', '红边面积', '蓝边位置', '黄边位置', '红边位置','蓝边位置反射率','黄边位置反射率','红边位置反射率','SDr/SDb','SDr/SDy','(SDr-SDb)/(SDr+SDb)','(SDr - SDy) / (SDr + SDy)']).T
		data2.to_excel(output_path+'/'+self.file_name+'_'+"three_edge.xlsx")

if __name__ == '__main__':
	parse = argparse.ArgumentParser ( description='add svc file and output the result' )
	parse.add_argument ( '-f', '--folder', help='the SVC EXCEL FILE' )


	args = parse.parse_args ()
	water_path = args.folder

	print('START RUNNING....')


	start=time.clock()


	workbook=xlrd.open_workbook(water_path)
	names=workbook.sheet_names()
	for name in names:
		spe = Spectral(water_path,name)
		spe.derivative()

	print('WRITING...')
	end=time.clock()
	print('read:%fs' %(end-start))

#python HS_02.py -f J:\2018玉米数据\玉米2018原始数据\qxym_2018\20180605\indoor_SVC