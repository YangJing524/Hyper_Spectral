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
from sklearn.model_selection import train_test_split, GridSearchCV
from pylab import mpl
import csv



plt.rc('font', family='Times New Roman',size=18)
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

print(__doc__)
start = time.clock()
starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s' % time.ctime())
print('-' * 80)

data_path = r'D:\Data\Python\Hyper_Spectral\训练集\多变量\SPAD-多变量.xlsx'
full_path = os.path.split(data_path)[0]

HS_path = full_path

# 该函数用于在y_test与y_predict重新计算OLS回归，用于绘图展示
def reg(X, y):
    reg = LinearRegression().fit(X, y)
    coef = round(reg.coef_[0][0], 4)
    intercept = round(reg.intercept_[0], 3)
    model = str('y=') + str(coef) + 'x+' + str(intercept)
    y_predict = reg.predict(X)
    delta = abs(y - y_predict)
    data = pd.Series(delta.reshape(1, -1)[0])

    index = data.index
    return index, model, coef, intercept, y_predict


def main():
    # 对应各个生育期
    data1 = pd.read_excel(data_path, sheet_name='1st')
    data2 = pd.read_excel(data_path, sheet_name='2nd')
    data3 = pd.read_excel(data_path, sheet_name='3rd')
    data4 = pd.read_excel(data_path, sheet_name='4th')
    data5 = pd.read_excel(data_path, sheet_name='5th')
    data6 = pd.read_excel(data_path, sheet_name='All')
    l = [data1, data2, data3, data4, data5, data6]

    # Load dataset

    def fit_and_draw():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        periods = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        for index, ax in enumerate(axes.ravel()):
            LR = LinearRegression()
            data = l[index]

            # X =np.vstack((data['光谱'], data['植被指数'], data['三边参数'])).transpose()
            # X=np.array(data['三边参数']).reshape(-1,1)
            X=data.iloc[:, 1:].values
            y = data['SPAD'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
            #k = 8
            reg_LR = LR.fit(X_train, y_train)

            #params = {'n_jobs': [1]}
            #reg_LR = GridSearchCV(LR, param_grid=params,cv=5)
            #reg_LR.fit(X_train,y_train)
            #建模步骤
            print(reg_LR.coef_)
            coef1= round(reg_LR.coef_[0], 2)
            coef2 = round(reg_LR.coef_[1], 2)
            coef3= round(reg_LR.coef_[2], 2)
            intercept = round(reg_LR.intercept_, 2)
            y_train_pre=reg_LR.predict(X_train)
            R_2 = r2_score(y_train,y_train_pre)
            RMSE1=round(np.sqrt(mean_squared_error(y_train,y_train_pre)),2)
            #计算验证集的R2和RMSE
            y_pre = reg_LR.predict(X_test)
            R2_2 = r2_score(y_test,y_pre)
            RMSE2 = round(np.sqrt(mean_squared_error(y_test, y_pre)), 3)
            # RE = round(np.mean(abs((y_pre - y_test)) / y_test), 3)

            print('Current parameter R2 is %s' % R_2)
            model = str(r'y = '+str(intercept)+'+ '+ str(coef1) + '*X1 +'+str(coef2) + '*X2 +'+str(coef3) + '*X3')
            #把方程式打印在excel里
            my_data = [{'periods': periods[index], 'model': model}, {'R2': R_2}, {'RMSE': RMSE1}]
            headers = ['periods', 'model', 'R2', 'RMSE']
            with open(os.path.join(HS_path, '_ols.csv'), 'a', newline='') as f:
                writer = csv.DictWriter(f, headers)
                writer.writeheader()
                writer.writerows(my_data)
            # plot

            #散点图的设置
            ax.scatter(y_test, y_pre, s=10, marker='o',color ='black')
            # 实测值与预测值再次建立回归模型，再次计算新模型的决定系数、rmse、绝对误差
            _, model, coef1, intercept1, y_pre_new = reg(np.array(y_test).reshape(-1,1), np.array(y_pre).reshape(-1,1))

            #验证集出图
            #line = np.linspace(25, 65, 5) * coef1 + intercept1
            #ax.plot(np.linspace(25, 65, 5), line, label=r'$R^2=$' + str(round(R_2,2)) + '\n' + 'RMSE=' + str(round(RMSE2, 2)),color='grey')
            line_2 = np.linspace(25, 65, 5)
            ax.plot(line_2,line_2,label=r'$R^2=$' + str(round(R2_2,2))+'\n'+'RMSE='+str(round(RMSE2,2)),linewidth=1,linestyle="--",color='grey')
            ax.legend(loc=0, edgecolor='white')
            ax.set_xlabel('Measured  LCC' + '\n' + '%s' % periods[index])
            ax.set_ylabel('Predicted  LCC')
            ax.set_xticks(np.linspace(25, 65, 5))
            ax.set_xticks(np.linspace(25, 65, 5))
            ax.set_xlim(25, 65)
            ax.set_ylim(25, 65)
            ax.set_aspect(1)
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
        plt.subplots_adjust(top=0.98,bottom=0.1,left=0.01,right=1,wspace=0.00, hspace=0.3)
        plt.savefig(os.path.join(HS_path, 'OLS' ),dpi=300)


    fit_and_draw( )


main()

end = time.clock()
print('Running:%fs' % (end - start))
print('DONE...')

















