r'''
	自行整理数据 参考如下格式整理Excel，并放入 以日期命名的sheet 中，必须将比值参数写为其他形式
	SPAD	NDVI	WI	VOG1	NDWI	（WI-NDWI）
0.091192753	0.747816054	1.046051303	1.3417481	0.055930666	18.70264338
0.080065359	0.747029793	1.048070413	1.308156232	0.054341737	19.28665657

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
plt.rc('font', family='Times New Roman')

print(__doc__)
start = time.clock()
starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s'%time.ctime())
print('-'*80)





data_path = r'D:\实验\代码\Hyper_Spectral\训练集\多变量\SPAD-多变量.xlsx'
full_path=os.path.split(data_path)[0]


HS_path=full_path

def reg(X,y):
		reg=LinearRegression().fit(X,y)
		coef=round(reg.coef_[0][0],2)
		intercept=round(reg.intercept_[0],2)
		model=str('y=')+str(coef)+'x+'+str(intercept)
		y_predict=reg.predict(X)
		delta=abs(y-y_predict)
		data=pd.Series(delta.reshape(1,-1)[0])
		index=data[data<4].index
		return index,model,coef,intercept,y_predict

def main():
	#对应各生育期
	data1 = pd.read_excel(data_path, sheet_name='1st')
	data2 = pd.read_excel(data_path, sheet_name='2nd')
	data3 = pd.read_excel(data_path, sheet_name='3rd')
	data4 = pd.read_excel(data_path, sheet_name='4th')
	data5 = pd.read_excel(data_path, sheet_name='5th')
	data6 = pd.read_excel(data_path, sheet_name='All')
	l=[data1,data2,data3,data4,data5, data6]

	#Load dataset

	def fit_and_draw(name):
		i=name
		fig,axes=plt.subplots(2,3,figsize=(15,10),dpi=300)

		periods=['(a)','(b)','(c)','(d)','(e)','(f)']
		for index,ax in enumerate(axes.ravel()):

			LR=LinearRegression()
			data=l[index]
			print('Current parameter is %s'%i)
			X=np.array(data[i]).reshape(-1,1)
			y = np.array ( data['SPAD'] ).reshape ( -1, 1 )
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3)

			#建模集步骤
			reg_LR=LR.fit(X_train,y_train)
			coef=round(reg_LR.coef_[0][0],2)
			intercept=round(reg_LR.intercept_[0],2)
			y_train_pre=reg_LR.predict ( X_train )
			R_2 = r2_score ((y_train, y_train_pre ),2)
			RMSE1 = round(np.sqrt(mean_squared_error(y_train,y_train_pre)),2)

			model=str(r'y = '+str(coef)+'x+'+str(intercept))

			#验证集的步骤
			y_pre = reg_LR.predict ( X_test )
			RMSE= round ( np.sqrt(mean_squared_error ( y_test, y_pre ) ),2)
			r_2 = round (( r2_score (y_test, y_pre )),2)
			#RE = round (np.mean(abs((y_pre - y_test))/y_test ),2)


			#with open ( os.path.join(HS_path,'%s_model_01.txt'%i), 'a+' ) as f:
				#f.writelines('当前参数为：'+str(i)+periods[index]+'\n')
				#f.writelines ( u'模型:' +model + '\n' )
				#f.writelines ( u'模型建立的时 R2:' +str(R_2) + '\n' )
				#f.writelines ( 'r_2:' + str ( r_2) + '\n' )
				#f.writelines ( 'RMSE:' + str (RMSE1) + '\n' )
				#f.writelines ( 'RE:' + str ( RE) + '\n' )
				#f.writelines ( '-'*50 + '\n' )

			my_data=[{'periods':periods[index],'model':model},{'R2':R_2},{'RMSE':RMSE1}]  #,{'RE%':RE*100}
			headers=['periods','model','R2','RMSE']
			with open(os.path.join(HS_path, '%s_ols.csv'%i), 'a', newline='') as f:
				writer=csv.DictWriter(f,headers)
				writer.writeheader()
				writer.writerows(my_data)
			#plot

			ax.scatter(y_test,y_pre,s=15,marker='o',color='blue')
			#实测值与预测值再次建立回归模型，再次计算新模型的决定系数、rmse、绝对误差
			_, model, coef1, intercept1,y_pre_new = reg ( y_test, y_pre )
			#rmse = round(np.sqrt(mean_squared_error(y_test,y_pre)),2)




			line=np.linspace(25,65,5)*coef1+intercept1
			ax.plot(np.linspace(25,65,5),line,label=r'$R^2=$'+str((r_2),2)+'\n'+'RMSE='+str(round(RMSE,2)),color='white')

			ax.set_xlabel('Measured value'+'\n'+'%s'%periods[index])
			ax.set_ylabel('Predicted value')
			line_2= np.linspace(25,65,5)
			ax.plot ( line_2,line_2,'--')
			ax.legend(loc=4,edgecolor='white')
			ax.set_xticks(np.linspace(25,65,5))
			ax.set_xticks(np.linspace(25,65,5))
			ax.set_xlim(25,65,5)
			ax.set_ylim(25,65,5)
			#ax.spines['top'].set_visible(True)
			#ax.spines['right'].set_visible(True)

		plt.savefig(os.path.join(HS_path,'./%s.png'%i), dpi=300)
		plt.subplots_adjust(top=1, bottom=0.1, left=0.05, right=0.98, wspace=0.25, hspace=0.25)

	col=data1.columns[1:]

	for j in range(len(col)):
		i=str(col[j])
		fit_and_draw(i)

main()

#end = time.clock()
#print('Running:%fs' % (end - start))
print('DONE...')

















