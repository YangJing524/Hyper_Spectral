#coding=cp936
u'''
	START:2019-03-06
	AUTHOR:YangJing
	USE:
		SVC���Աȷ������׸������ڹ������߲���
	INPUT:
	OUTPUT:
		svc.png
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import time

print(__doc__)
start=time.clock()
print('Current time:%s'%time.ctime())
print('-'*80)
parse = argparse.ArgumentParser ( description='add svc file and output the result' )
parse.add_argument ( '-svc', '--svc', help='the SVC EXCEL FILE' )

args = parse.parse_args ()
s_path = args.svc
full_path=rr'J:\��������\output\water'
all_files=os.listdir(s_path)

try:
	os.makedirs(os.path.join(full_path,'HS_05'))
except:
	pass
HS_path=os.path.join(full_path,'HS_05')

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


#����Լ������ͬһ�����ݰ��������¡�������3��excel�ļ���
def draw_svc():
	fig,axes=plt.subplots(figsize=(8,5))

	#plt.xlabel(all_files)


	#�˴������ļ��У��Լ���Ҫ���Ƶĵ�ͼaxes0
	def get(linestyle, label):
		files=['S.xlsx']

		S=pd.read_excel(os.path.join(svc_path,files[0]))
		S=S[S.columns[1:850]]

		axes.plot(S.iloc[20], linestyle=linestyle, label=label)
		axes.set_xlim(350,1000)
		axes.set_ylim(0,0.6)
		axes.spines['top'].set_visible ( False )
		axes.spines['right'].set_visible ( False )
		axes.set_xlabel('���� WaveLength/nm', {'family':'SimHei', 'size':10})
		axes.set_ylabel ( '������ Reflectance', {'family':'SimHei', 'size':10})

	linestyle =['-','--','-.',':']
	labels = ['�ν��� Elongation stage', '������ Tasseling stage', '������ Milk stage', '������ Maturing stage']

	for id,file in tqdm(enumerate(all_files)):
		svc_path=os.path.join(s_path,all_files[id])
		get(linestyle=linestyle[id], label=labels[id])

	plt.legend(loc='best', prop={'family':'SimHei', 'size':10}, edgecolor='white')
	plt.savefig(os.path.join(HS_path,'./svc.png'))
	print('DONE...')


draw_svc()

end = time.clock()
print('Running:%fs' % (end - start))