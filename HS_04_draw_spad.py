import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd
import os
import argparse
import time

print(__doc__)
start=time.clock()
print('Current time:%s'%time.ctime())
print('-'*80)


parse = argparse.ArgumentParser ( description='add spad file and output the result' )

parse.add_argument ( '-spad', '--spad', help='the SPAD EXCEL FILE' )

args = parse.parse_args ()

spad_path = args.spad
full_path=r'D:\实验\代码\Hyper_Spectral\训练集\SPAD原始.xlsx'

try:
	os.makedirs(os.path.join(full_path,'HS_04'))
except:
	pass
HS_path=os.path.join(full_path,'HS_04')
print(HS_path)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def draw_spad():
	path=HS_path
	workbook=xlrd.open_workbook(path)
	names=workbook.sheet_names()
	sheet1st=pd.read_excel(path,sheet_name=names[0],names=[names[0]],header=None)
	sheet2nd=pd.read_excel(path,sheet_name=names[1],names=[names[1]],header=None)
	sheet3rd=pd.read_excel(path,sheet_name=names[2],names=[names[2]],header=None)
	sheet4th=pd.read_excel(path,sheet_name=names[3],names=[names[3]],header=None)
	sheet5th=pd.read_excel(path,sheet_name=names[4],names=[names[4]],header=None)
	sheetAll=pd.read_excel(path,sheet_name=names[5],names=[names[5]],header=None)
	spad=pd.concat([sheet1st, sheet2nd, sheet3rd, sheet4th,sheet5th,sheetAll],axis=1)
	#spad.columns = ['拔节期 Elongation stage', '抽雄期 Tasseling stage', '乳熟期 Milk stage', '完熟期 Maturing stage']
	spad.columns = names
	sns.boxplot(data=spad)
	sns.swarmplot(data=spad)
	ax=plt.gca()
	ax.spines['top'].set_color('none')
	ax.spines['right'].set_color('none')

	plt.xlabel('时期(periods)')
	plt.ylabel('叶片叶绿素含量(leaf chlorophyll content)')
	plt.savefig(HS_path+'./'+'SPAD_01.png')

	with open(HS_path+'./'+'percent.txt','w') as f:
		for i in spad.columns:
			Q1 = np.percentile(spad[i].dropna(), 25)
			Q3 = np.percentile(spad[i].dropna(), 75)
			IQR=Q3-Q1
			outlier_step=1.5*IQR
			outlier_value_top=Q1-outlier_step
			outlier_value_bottom = Q3 + outlier_step
			out_value=(spad[spad[i]<outlier_value_top])|(spad[spad[i]>outlier_value_bottom])
			index=out_value.index
			value=spad[i][out_value.index]
			f.writelines('时期：%s'%i+'\n')
			f.writelines('序号：%s' % index + '\n')
			f.writelines('值：%s' % value + '\n')


print('DONE...')
end = time.clock()
print('Running:%fs' % (end - start))