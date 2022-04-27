r'''
    自行整理数据 参考如下格式整理Excel，并放入 以编号命名的 SHEET 中
            SPAD	R
        0	41.1	0.1991
        1	26.3	0.2343
        2	36.8	0.208
        3	33.6	0.2043
        4	28.2	0.1572
        5	27.8	0.1991
        6	39.4	0.2108
        7	33      0.168
        8	34.8	0.2129
        9	31.8	0.1942
        10	38.1	0.2236
        
    该脚本由于 SPAD 与相关系数做普通回归分析
    主要包含以下几种：
                        一元线性：y=ax+b
                        二次多项式：y=ax2+bx+c
	
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

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

print(__doc__)
start = time.clock()
starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s'%time.ctime())
print('-'*80)





data_path = r'J:\论文数据\output\ordinary_regress_spad\原始光谱特征波段(去缺失值)\regress.xlsx'
#缺失值索引,来源于HS_04.PY , percent.txt, 通过四分位计算缺失值
drop_list_01=list(map(lambda x:x+1,[1, 4, 5, 11, 217]))
drop_list_02=list(map(lambda x:x+1,[2,  15,  16,  22,  38,  40,  54,  55,  58,  60,  62,  65,  67,68, 125, 130, 131, 132, 133, 145, 149, 154, 157, 160]))
drop_list_03=list(map(lambda x:x+1,[24,  25,  26,  27,  30,  31,  32,  33,  34,  35,  38,  39,  41,42,  48,  54,  55,  58,  61,  85,  87,  88,  89,  91,  93,  96,97, 100, 101, 123]))
drop_list_04=list(map(lambda x:x+1,[59, 60, 121, 123, 126, 175, 184, 191, 207, 210, 227, 364, 421, 463,519]))

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
	data1 = pd.read_excel(data_path, sheet_name='0605', skiprows=drop_list_01)
	data2 = pd.read_excel(data_path, sheet_name='0622', skiprows=drop_list_02)
	data3 = pd.read_excel(data_path, sheet_name='0717', skiprows=drop_list_03)
	data4 = pd.read_excel(data_path, sheet_name='0826', skiprows=drop_list_04)
	l=[data1,data2,data3,data4]
	

	fig,axes=plt.subplots(2,2,figsize=(8,8))

	for index,ax in enumerate(axes.ravel()):
		data=l[index]
		y = np.array ( data['SPAD'] ).reshape ( -1, 1 )
		i=str(data.columns[-1])
		LR=LinearRegression()
		
		print('current parameter is %s'%i)
		X=np.array(data[i]).reshape(-1,1)
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4)

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

		with open ( os.path.join(HS_path,'%s_model_01.txt'%i), 'a+' ) as f:
			f.writelines('当前参数为：'+str(i)+'\n')
			f.writelines ( u'模型:' +model + '\n' )
			f.writelines ( u'模型建立的时 R2:' +str(R_2) + '\n' )
			f.writelines ( 'r_2:' + str ( r_2) + '\n' )
			f.writelines ( 'rmse:' + str ( rmse) + '\n' )
			f.writelines ( 'RE:' + str ( RE) + '\n' )
			f.writelines ( '--'*50 + '\n' )

		#plot

		ax.scatter(y_test,y_pre,s=10,marker='*')
		#实测值与预测值再次建立回归模型，再次计算新模型的决定系数、rmse、绝对误差
		_, model, coef1, intercept1,y_pre = reg ( y_test, y_pre )

		rmse = round ( np.sqrt ( mean_squared_error ( y_test, y_pre ) ), 5 )
		r_2 = round ( r2_score (y_test, y_pre ), 5 )
		RE = round ( np.mean ( abs ( (y_pre - y_test) ) / y_test ), 5 )

		# with open ( os.path.join(HS_path,'%s_model_02.txt'%i), 'a+' ) as f:
			# f.writelines('当前参数为：'+str(i)+'\n')
			# f.writelines ( u'模型:' +model + '\n' )
			# f.writelines ( 'r_2:' + str ( r_2) + '\n' )
			# f.writelines ( 'rmse:' + str ( rmse) + '\n' )
			# f.writelines ( 'RE:' + str ( RE) + '\n' )
			# f.writelines ( '--'*50 + '\n' )

		line=np.arange(0,np.max(y_test))*coef1+intercept1

		ax.plot(line,label=model+'\n'+r'$R^2=$'+str(r_2))
		ax.set_xlabel('实测值 Measured value'+'\n'+'%s'%i)
		ax.set_ylabel('预测值 Predicted value')
		line_2= np.arange ( 0, np.max ( y_test ) )
		ax.plot ( line_2,'--')
		ax.legend(loc=4,edgecolor='white')
		ax.set_xlim(20,60)
		ax.set_ylim(20,60)
		ax.spines['top'].set_visible ( False )
		ax.spines['right'].set_visible ( False )

	plt.savefig(os.path.join(HS_path,'./regress.png'))


main()

end = time.clock()
print('Running:%fs' % (end - start))
print('DONE...')

















