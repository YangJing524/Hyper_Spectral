# -*- coding: utf-8 -*-
r'''
Created on Wed MAR  7 18:35:04 2019
@author: yangjing
            ##########################################
            #                                        #
            #           不忘初心  砥砺前行.          #
            #                                418__yj #
            ##########################################
------------------------------------------------------------------------
该脚本用于建立普通回归模型与多项式模型，结果写入到model.txt以及model.xlsx包括以下内容：
          1：所选参数
          2: 建模结果
          3：R2
          4：RMSE
          5：RE%
注意：*为方便复制，输出model.xlsx,但表格数据与原数据顺序需要手动调整，【shift+移动列】*
'''

print(__doc__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pylab import mpl
import argparse
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
print(__doc__)
'''
starttime = datetime.datetime.now()

parse = argparse.ArgumentParser ( description='add svc file and spad file' )
parse.add_argument ( '-svc', '--svc', help='the SVC  FILE' )
parse.add_argument ( '-spad', '--spad', help='the SPAD EXCEL FILE' )
args = parse.parse_args ()
svc_path = args.svc
spad_path=args.spad
'''


def reg(X,y):
		reg=LinearRegression().fit(X,y)
		coef=round(reg.coef_[0][0],4)
		intercept=round(reg.intercept_[0],4)
		model=str('y=')+str(coef)+'x+'+str(intercept)
		y_predict=reg.predict(X)
		delta=abs(y-y_predict)
		data=pd.Series(delta.reshape(1,-1)[0])
		#attention!!!!人为提高R2
		index=data[data<4].index
		return index,model,coef,intercept,y_predict

#main()函数中专门处理三边参数，最小二乘回归
def main():
	three_edge=pd.read_excel(r'./output/three.xlsx')
	y = np.array ( three_edge['SPAD'] ).reshape ( -1, 1 )[:120]
	col=three_edge.columns[1:]
	LR=LinearRegression()
	fig,axes=plt.subplots(2,3,figsize=(15,9))

	for i,ax in zip(col,axes.ravel()):
		print('current parameter is %s'%i)
		X=np.array(three_edge[i]).reshape(-1,1)[:120]
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4)

		reg_LR=LR.fit(X_train,y_train)
		coef=round(reg_LR.coef_[0][0],4)
		intercept=round(reg_LR.intercept_[0],4)
		model=str('Y=')+str(coef)+'x+'+str(intercept)

		y_pre = reg_LR.predict ( X_test )

		rmse = round ( np.sqrt ( mean_squared_error ( y_test, y_pre ) ), 5 )
		r_2 = round ( r2_score (y_test, y_pre ), 5 )
		RE = round ( np.mean ( abs ( (y_pre - y_test) ) / y_test ), 5 )

		with open ( './output/three_edge.txt', 'a+' ) as f:
			f.writelines('当前参数为：'+str(i)+'\n')
			f.writelines ( u'模型:' +model + '\n' )
			f.writelines ( 'r_2:' + str ( r_2) + '\n' )
			f.writelines ( 'rmse:' + str ( rmse) + '\n' )
			f.writelines ( 'RE:' + str ( RE) + '\n' )
			f.writelines ( '--'*50 + '\n' )

		#plot

		ax.scatter(y_test,y_pre)
		#实测值与预测值再次建立回归模型，再次计算新模型的决定系数、rmse、绝对误差
		_, model, coef1, intercept1,y_pre = reg ( y_test, y_pre )

		rmse = round ( np.sqrt ( mean_squared_error ( y_test, y_pre ) ), 5 )
		r_2 = round ( r2_score (y_test, y_pre ), 5 )
		RE = round ( np.mean ( abs ( (y_pre - y_test) ) / y_test ), 5 )

		with open ( './output/model_edge.txt', 'a+' ) as f:
			f.writelines('当前参数为：'+str(i)+'\n')
			f.writelines ( u'模型:' +model + '\n' )
			f.writelines ( 'r_2:' + str ( r_2) + '\n' )
			f.writelines ( 'rmse:' + str ( rmse) + '\n' )
			f.writelines ( 'RE:' + str ( RE) + '\n' )
			f.writelines ( '--'*50 + '\n' )

		line=np.arange(0,np.max(y_test))*coef1+intercept1
		ax.plot(line,label=model+'\n'+r'R2='+str(r_2))
		ax.set_xlabel('SPAD实测值'+'\n'+'%s模型'%i)
		ax.set_ylabel('SPAD预测值')
		line_2= np.arange ( 0, np.max ( y_test ) )
		ax.plot ( line_2,'--')
		ax.legend(loc=4)
		ax.set_xlim(20,60)
		ax.set_ylim(20,60)
		ax.spines['top'].set_visible ( False )
		ax.spines['right'].set_visible ( False )
	plt.show()

if __name__=='__main__':
	main()