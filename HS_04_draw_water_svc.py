import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

start=time.clock()
print('Current time:%s'%time.ctime())
print('-'*80)

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

labels=['拔节期 Elongation stage', '抽雄期 Tasseling stage', '乳熟期 Milk stage', '完熟期 Maturing stage']
#['0605', '0622', '0717', '0826']
HS_path=r'J:\论文数据\output\water\HS_04'
data_path=r'J:\论文数据\output\water\Z_half.xlsx'

def draw_spad_svc():
	fig,axes=plt.subplots(2,2,figsize=(10,6.18))
	
	svc_0605=pd.read_excel(data_path, sheet_name='0605')
	svc_0605=svc_0605[svc_0605.columns[1:1500]]
	svc_0605=svc_0605.iloc[[14,43,35]].T
	svc_0605.columns=['3.9%','8.9%','13.7%']

	svc_0622=pd.read_excel(data_path, sheet_name='0622')
	svc_0622=svc_0622[svc_0622.columns[1:1500]]
	svc_0622=svc_0622.iloc[[7,21,12]].T
	svc_0622.columns=['7.8%','15.0%','22.5%']

	svc_0717=pd.read_excel(data_path, sheet_name='0717')
	svc_0717=svc_0717[svc_0717.columns[1:1500]]
	svc_0717=svc_0717.iloc[[24,49,8]].T
	svc_0717.columns=['17.1%','24.0%','29.4%']

	svc_0826=pd.read_excel(data_path, sheet_name='0826')
	svc_0826=svc_0826[svc_0826.columns[1:1500]]
	svc_0826=svc_0826.iloc[[51,58,52]].T
	svc_0826.columns=['15.6%','25.2%','39.3%']
	#plt.xlabel(all_files)


	#此处传入文件，以及需要绘制的底图axes0
	def get(data, title, ax,s1,s2):
		
		axes=ax

		col=data.columns
		axes.plot(data[col[0]],linestyle='-.',label=col[0])

		axes.plot(data[col[1]],linestyle='--',label=col[1])

		axes.plot(data[col[2]],linestyle=':',label=col[2])
		
		axes.set_xlim(350,1800)
		axes.set_ylim(0,0.6)
		axes.spines['top'].set_visible ( False )
		axes.spines['right'].set_visible ( False )
		axes.set_xlabel(s1,s2)
		axes.set_ylabel ( '反射率 Reflectance', {'family':'SimHei', 'size':10})
		
		axes.legend(loc='best', prop={'family':'SimHei', 'size':9}, edgecolor='white')
		axes.set_title(title, {'family':'SimHei', 'size':10})
	s1='波长 WaveLength/nm' 
	s2={'family':'SimHei', 'size':10}
	get(svc_0605,labels[0],axes[0][0],'',{})
	get(svc_0622,labels[1],axes[0][1],'',{})
	get(svc_0717,labels[2],axes[1][0],s1,s2)
	get(svc_0826,labels[3],axes[1][1],s1,s2)



	plt.savefig(os.path.join(HS_path,'./svc_water.png'))
	print('DONE...')

draw_spad_svc()

end = time.clock()
print('Running:%fs' % (end - start))