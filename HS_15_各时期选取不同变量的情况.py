r'''
    自行整理数据 参考如下格式整理Excel，并放入 以日期命名的sheet 中，必须将比值参数写为其他形式
	SPAD	NDVI	WI	VOG1	NDWI	（WI-NDWI）
0.091192753	0.747816054	1.046051303	1.3417481	0.055930666	18.70264338
0.080065359	0.747029793	1.048070413	1.308156232	0.054341737	19.28665657
0.080764488	0.729497354	1.049204052	1.221287842	0.055720883	18.8296381
0.055387819	0.74600639	1.045696277	1.380875638	0.056603774	18.47396757
0.068351361	0.756081194	1.05111755	1.375104661	0.060047379	17.50480324
0.101584751	0.760787825	1.046881814	1.394096166	0.053751765	19.47623157
0.062190744	0.752103787	1.049978364	1.354010387	0.056636217	18.5389918
0.068851859	0.744299357	1.04427334	1.256709452	0.052392344	19.93179251
0.086923254	0.757803563	1.05003026	1.259241126	0.058729204	17.87918419
0.066938127	0.733067066	1.045379196	1.239851485	0.054616653	19.14030136

        
    该脚本由于 water 与相关系数做普通回归分析
    主要包含以下几种：
                        一元线性：y=ax+b
	
	(a)/(b)/(c)/(d)分别代表四个生育期
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import time
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pylab import mpl
import csv
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

print(__doc__)
start = time.clock()
starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s'%time.ctime())
print('-'*80)





data_path = r'J:\论文数据\output\ordinary_regress_water\遥感参数\water.xlsx'
full_path=os.path.split(data_path)[0]


HS_path=full_path

def reg(X,y):
		reg=LinearRegression().fit(X,y)
		coef=round(reg.coef_[0][0],4)
		intercept=round(reg.intercept_[0],3)
		model=str('y=')+str(coef)+'x+'+str(intercept)
		y_predict=reg.predict(X)
		delta=abs(y-y_predict)
		data=pd.Series(delta.reshape(1,-1)[0])
		index=data[data<4].index
		return index,model,coef,intercept,y_predict

def main():
	#对应玉米的四个生育期
	data1 = pd.read_excel(data_path, sheet_name='0605')
	data2 = pd.read_excel(data_path, sheet_name='0622')
	data3 = pd.read_excel(data_path, sheet_name='0717')
	data4 = pd.read_excel(data_path, sheet_name='0826')

	
	#Load dataset
	col_0605=['SDy','SDr','Ry','Rr','SDr#SDy']
	col_0622=['Dr','Ry']
	col_0717=['Dr','Ry']
	col_0826=['Db','SDr#SDb','(SDr - SDy) # (SDr + SDy)']
	
	periods=['0605','0622','0717','0826']
	
	
	
	def fit_and_draw(data,cols,period):
	
		try:
			os.makedirs(os.path.join(full_path,period))
		except:
			pass
		HS_path=os.path.join(full_path,period)
		all_data=[]
		for index,col in enumerate(cols):
		
			fig,axes=plt.subplots(1,1,figsize=(4,4))
			LR=LinearRegression()
			data=data
			print('Current parameter is %s'%col)
			X=np.array(data[col]).reshape(-1,1)
			y = np.array ( data['water'] ).reshape ( -1, 1 )
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3)

			reg_LR=LR.fit(X_train,y_train)
			coef=round(reg_LR.coef_[0][0],4)
			intercept=round(reg_LR.intercept_[0],3)
			y_train_pre=reg_LR.predict ( X_train )
			R_2=round ( r2_score (y_train, y_train_pre ), 3 )
			
			model=str(r'y = '+str(coef)+'x+'+str(intercept))

			y_pre = reg_LR.predict ( X_test )

			rmse = round ( np.sqrt ( mean_squared_error ( y_test, y_pre ) ), 3 )
			r_2 = round ( r2_score (y_test, y_pre ), 3 )
			RE = round ( np.mean ( abs ( (y_pre - y_test) ) / y_test ), 3 )
			
			
			with open ( os.path.join(HS_path,'%s_%s.txt'%(period,col)), 'a+' ) as f:
				f.writelines('当前参数为：'+str(col)+':'+period+'\n')
				f.writelines ( u'模型:' +model + '\n' )
				f.writelines ( u'模型建立的时 R2:' +str(R_2) + '\n' )
				f.writelines ( 'r_2:' + str ( r_2) + '\n' )
				f.writelines ( 'rmse:' + str ( rmse) + '\n' )
				f.writelines ( 'RE:' + str ( RE) + '\n' )
				f.writelines ( '-'*50 + '\n' )
				
			my_data=[{'col':col},{'model':model},{'R2':R_2},{'r2':r_2},{'RMSE':rmse},{'RE%':RE*100}]
			all_data.append(my_data)

			#plot
			ax=axes
			ax.scatter(y_test,y_pre,s=10,marker='*')
			#实测值与预测值再次建立回归模型，再次计算新模型的决定系数、rmse、绝对误差
			_, model, coef1, intercept1,y_pre_new = reg ( y_test, y_pre )

			# with open ( os.path.join(HS_path,'%s_model_02.txt'%i), 'a+' ) as f:
				# f.writelines('当前参数为：'+str(i)+'\n')
				# f.writelines ( u'模型:' +model + '\n' )
				# f.writelines ( 'r_2:' + str ( r_2) + '\n' )
				# f.writelines ( 'rmse:' + str ( rmse) + '\n' )
				# f.writelines ( 'RE:' + str ( RE) + '\n' )
				# f.writelines ( '--'*50 + '\n' )

			line=np.linspace(0,0.35,50)*coef1+intercept1
			
			ax.plot(np.linspace(0,0.35,50),line,label=model+'\n'+r'$R^2=$'+str(r_2))
			ax.set_xlabel('实测值 Measured value')
			ax.set_ylabel('预测值 Predicted value')
			line_2= np.linspace(0,0.35,50)
			ax.plot ( line_2,line_2,'--')
			ax.legend(loc=4,edgecolor='white')
			ax.set_xticks(np.linspace(0,0.35,8))
			ax.set_xticks(np.linspace(0,0.35,8))
			ax.set_xlim(0,0.35)
			ax.set_ylim(0,0.35)
			ax.spines['top'].set_visible ( False )
			ax.spines['right'].set_visible ( False )

			plt.savefig(os.path.join(HS_path,'./%s_%s.png'%(period,col)))
		
		print(all_data)
		headers=['col','model','R2','r2','RMSE','RE%']
		with open(os.path.join(HS_path, '%s.csv'%(period)), 'a', newline='') as f:
			writer=csv.DictWriter(f,headers)
			writer.writeheader()
			for DATA in all_data:
				writer.writerows(DATA)
	#fit_and_draw(data1,col_0605,periods[0])
	fit_and_draw(data2,col_0622,periods[1])
	fit_and_draw(data3,col_0717,periods[2])
	fit_and_draw(data4,col_0826,periods[3])



main()

end = time.clock()
print('Running:%fs' % (end - start))
print('DONE...')

















