'''
	作者：418--杨景--
	时间：20180825
	说明：该脚本基于python3第三方库pandas构建，用于批量处理实验中的多酚数据
	调试联系：2508204932@qq.com
'''
import re
import os
import pandas as pd
import numpy as np
import time

class DX(object):
	def __init__(self,input_path,output_path):
		self.input_path=input_path
		self.output_path=output_path
		self.headers=['#Group','Chl','Chl_sd','Flav','Flav_sd','Anth','Anth_sd','NBI','NBI_sd']
	'''
		绝对路径处理函数:path_pro()
	'''
	def main(self):
		pass

	def open_file(self,file):
		full_path=self.input_path+'//'+file
		return full_path

	def start(self):
		print('程序正在运行...')
		'''中文路径名：Error:Initializing from file failed'''
		file=os.listdir(self.input_path)
		for filename in file:
			file_=self.open_file(filename)
			print('当前处理文件：%s'%file_)
			f=open(file_)
			pd.set_option('display.max_colwidth', 1000)
			raw_data=pd.read_csv(f,skiprows=4,header=None)
			l = [line_num for line_num in range(len(raw_data)) if str(raw_data.iloc[line_num]).startswith('0    #Group')]
			ll=[i+1 for i in l]
			all_data=[]
			for i in ll:
				all_data.append(raw_data.iloc[i].reset_index())
			split_data = [str(all_data[i]).split(';') for i in range(len(all_data))]
			for i in range(len(split_data)):
				split_data[i][0]=split_data[i][0][-3:]
			export_data=pd.DataFrame(split_data,columns=self.headers)
			self.save_file(export_data,filename)


	def save_file(self,data,filename):
		save_path=self.output_path+'//'+filename[:-4]+'.xlsx'
		print('当期保存路径：%s'%save_path)
		print()
		data.to_excel(save_path)


if __name__=='__main__':
	start_time=time.clock()
	print('---------------------程序处理中切勿将输入路径与保存文件路径设为相同路径-------Build by:@Anaconda3.Inc--------')
	print('START TIME:%s'%(time.ctime()))
	print()
	print()
	'''
		----------------------输入要处理的所有含有多酚数据的文件夹-----------------------
	'''

	#input_path=r'C:\Users\Administrator\Desktop\test'
	#output_path=r'C:\Users\Administrator\Desktop\out'

	input_path=input('请输入要处理的所有多酚csv文件的文件夹:')
	if os.path.exists(input_path)==False:
		print('文件夹不存在。。。')
		input_path=None
		input_path = input('请重新输入要处理的所有多酚csv文件的文件夹:')

	output_path=input('指定输出文件夹:')
	if os.path.exists(input_path)==False:
		print('文件夹不存在。。。')
	elif input_path==output_path:
		print('与输入文件夹相同，请重新输入！！！')
		output_path=None
		output_path = input('重新指定输出文件夹:')
	print('程序开始运行时间: %s' % time.ctime())
	run=DX(input_path,output_path)
	run.start()
	print('程序运行时间: %s' % (time.clock()-start_time))
