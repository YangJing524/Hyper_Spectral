'''
	knn :algorithm model
	TheilSen :algorithm model
	XGBoost : algorithm model
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import csv
import argparse
import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pylab import mpl
from sklearn.model_selection import KFold
from sklearn.linear_model import TheilSenRegressor

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

###路径初始化
data_path = r'J:\论文数据\output\algorithm_spad\MULITI_高光谱参数_去偏离值\regress_03.xlsx'
full_path=os.path.split(data_path)[0]
HS_path=full_path

def performance_metric(y_true, y_predict):
	""" Calculates and returns the performance score between 
	true and predicted values based on the metric chosen. """
	score = r2_score(y_true, y_predict)
	return score

def theilSen_model_k_fold(X,y):

	k_fold = KFold(n_splits=5)
	regressor = TheilSenRegressor(random_state=42)
	scoring_fnc = make_scorer(performance_metric)
	params={'random_state':[42]}
	grid = GridSearchCV(regressor, param_grid=params,scoring=scoring_fnc,cv=k_fold)

	# Fit the grid search object to the data to compute the optimal model
	theil_grid = grid.fit(X, y)

	 # Return the optimal model after fitting the data
	return theil_grid

def xgboost_model_k_fold(X,y):
	import xgboost as xgb
	scoring_fnc = make_scorer(performance_metric)
	k_fold = KFold(n_splits=5)
	'''XGBoost 调参心得：将部分已经调好的参数要固定，然后调整需要变化的参数'''
	parameters = {
		'max_depth': [2],
		'learning_rate':[0.05,0.1,0.2],
		'n_estimators': [150],
		'min_child_weight':[1,2,3,4],
		'silent': [1],
		'objective': ['reg:gamma'],
		'booster': ['gbtree'],
		'reg_alpha':[0],
		'reg_lambda':[1]
	}
	model = xgb.XGBRegressor (**parameters)
	grid_search = GridSearchCV ( model, parameters ,scoring=scoring_fnc,cv=k_fold)

	xgb_reg = grid_search.fit ( X, y)
	print(xgb_reg.best_params_)
	return xgb_reg
	
def knn_model_k_fold(X, y):
	""" Performs grid search over the 'max_depth' parameter for a 
	decision tree regressor trained on the input data [X, y]. """
	# Create cross-validation sets from the training data
	# cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
	k_fold = KFold(n_splits=5)
	
	# TODO: Create a decision tree regressor object
	regressor = KNeighborsRegressor()

	# TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
	params = {'n_neighbors':[2]}

	# TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
	scoring_fnc = make_scorer(performance_metric)
	# TODO: Create the grid search object
	grid = GridSearchCV(regressor, param_grid=params,scoring=scoring_fnc,cv=k_fold)

	# Fit the grid search object to the data to compute the optimal model
	knn_grid = grid.fit(X, y)

	 # Return the optimal model after fitting the data
	print(knn_grid.best_params_)
	return knn_grid
	



def reg(X,y):
		reg=LinearRegression().fit(X,y)
		coef=round(reg.coef_[0][0],4)
		intercept=round(reg.intercept_[0],3)
		model=str('y=')+str(coef)+'x+'+str(intercept)
		y_predict=reg.predict(X)
		delta=abs(y-y_predict)
		data=pd.Series(delta.reshape(1,-1)[0])
		#attention!!!!若人为提高R2，需接受 index 参数
		index=data[data<4].index
		return index,model,coef,intercept,y_predict

#main()函数传入参数为【1】训练好的模型 【2】model名称  
#返回不同模型的  R2，RMSE,RE,以及SPAD实测值与SPAD预测值的新的回归模型以及r2
def main(model,model_name,col):
	
	periods=['(a)','(b)','(c)','(d)']
	fig,axes=plt.subplots(2,2,figsize=(8,8))
	
	def fit_and_draw():
		for index,ax in enumerate(axes.ravel()):
			data=l[index]
			y = np.array ( data['SPAD'] ).reshape ( -1, 1 )
			
			print('current algorithm is %s '%model_name)
			X=np.array(data[col])
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4)

			Model = model(X_train,y_train)
			y_train_pre = Model.predict(X_train).reshape(-1,1)
			R_2=round ( r2_score (y_train_pre, y_train ), 3 )

			y_pre = Model.predict ( X_test ).reshape(-1,1)
			#print(y_pre)
			rmse = round ( np.sqrt ( mean_squared_error ( y_test, y_pre ) ), 3 )
			r_2 = abs(round ( r2_score (y_test, y_pre ), 3 ))
			RE = round ( np.mean ( abs ( y_test-y_pre ) / y_test ), 3 )
			RE = RE*100
			with open ( os.path.join(HS_path,'%s_model.txt'%model_name), 'a+' ) as f:
				f.writelines('当前参数为：'+model_name+'_'+periods[index]+'\n')
				#f.writelines ( u'模型:' +model + '\n' )
				f.writelines ( u'模型建立的时 R2:' +str(R_2) + '\n' )
				f.writelines ( 'r_2:' + str ( r_2) + '\n' )
				f.writelines ( 'rmse:' + str ( rmse) + '\n' )
				f.writelines ( 'RE:' + str ( RE) + '\n' )
				f.writelines ( '-'*50 + '\n' )

			#plots
			my_data=[{'periods':periods[index],'model':model},{'R2':R_2},{'r2':r_2},{'RMSE':rmse},{'RE%':RE*100}]
			headers=['periods','model','R2','r2','RMSE','RE%']
			with open(os.path.join(HS_path, '%s_model_ols.csv'%model_name), 'a', newline='') as f:
				writer=csv.DictWriter(f,headers)
				writer.writeheader()
				writer.writerows(my_data)
				
			ax.scatter(y_test,y_pre,s=10,marker='*')
			#实测值与预测值再次建立回归模型，再次计算新模型的决定系数、rmse、绝对误差
			_, model_, coef1, intercept1,y_pre = reg ( y_test, y_pre )

			rmse = round ( np.sqrt ( mean_squared_error ( y_test, y_pre ) ), 5 )
			r_2 = round ( r2_score (y_test, y_pre ), 5 )
			RE = round ( np.mean ( abs ( (y_pre - y_test) ) / y_test ), 5 )


			line=np.arange(0,np.max(y_test))*coef1+intercept1

			ax.plot(line,label=model_+'\n'+r'$R^2=$'+str(abs(r_2)))
			ax.set_xlabel('实测值 Measured value'+'\n'+'%s'%periods[index])
			ax.set_ylabel('预测值 Predicted value')
			line_2= np.arange ( 0, np.max ( y_test ) )
			ax.plot ( line_2,'--')
			ax.legend(loc=4,edgecolor='white')
			ax.set_xlim(20,60)
			ax.set_ylim(20,60)
			ax.spines['top'].set_visible ( False )
			ax.spines['right'].set_visible ( False )

		plt.savefig(os.path.join(HS_path,'./%s.png'%model_name))
		print('-'*50)
	
	fit_and_draw()


if __name__=='__main__':
	print(__doc__)
	start = time.clock()
	starttime = datetime.datetime.now()
	print('START RUNNING...')
	print('Current time:%s'%time.ctime())
	print('-'*80)
	drop_list_01=list(map(lambda x:x+1,[1, 4, 5, 11, 217]))
	drop_list_02=list(map(lambda x:x+1,[2,  15,  16,  22,  38,  40,  54,  55,  58,  60,  62,  65,  67,68, 125, 130, 131, 132, 133, 145, 149, 154, 157, 160]))
	drop_list_03=list(map(lambda x:x+1,[24,  25,  26,  27,  30,  31,  32,  33,  34,  35,  38,  39,  41,42,  48,  54,  55,  58,  61,  85,  87,  88,  89,  91,  93,  96,97, 100, 101, 123]))
	drop_list_04=list(map(lambda x:x+1,[59, 60, 121, 123, 126, 175, 184, 191, 207, 210, 227, 364, 421, 463,519]))
	#read data
	data1 = pd.read_excel(data_path, sheet_name='0605', skiprows=drop_list_01)
	data2 = pd.read_excel(data_path, sheet_name='0622', skiprows=drop_list_02)
	data3 = pd.read_excel(data_path, sheet_name='0717', skiprows=drop_list_03)
	data4 = pd.read_excel(data_path, sheet_name='0826', skiprows=drop_list_04)
	l=[data1,data2,data3,data4]
	
	col=data1.columns[1:]
	main(xgboost_model_k_fold,'xgboost',col)
	main(theilSen_model_k_fold,'theilSen',col)
	#main(knn_model_k_fold,'knn',col)
	
	end = time.clock()
	print('Running:%fs' % (end - start))
	print('DONE...')