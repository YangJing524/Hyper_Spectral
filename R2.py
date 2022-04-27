from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
df1 = pd.read_excel('D:\My_article\第二篇\DATA\VIs\原始VI-N.xlsx')


X = np.array(df1[['RVI']])#np.array(df1.iloc[:,1:])
Y =  np.array(df1[['N']])#np.array(df1.iloc[:,0:])
#plt.rc('font',family='STXihei',size=15)
#plt.scatter(X,Y,60,color='blue',marker='o',linewidths=3,alpha=0.4)
#plt.xlabel('NDVI')
#plt.ylabel('logN')
#plt.title('NDVI')
#plt.show()
reg = linear_model.LinearRegression()
reg.fit(X,Y)
#RMSE = reg.sqrt ( mean_squared_error (X,Y) ), 3
R_2 = reg.score (X,Y)
#print(reg.coef_,reg.intercept_)
print(R_2)
