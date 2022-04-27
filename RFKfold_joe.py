# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:25:17 2020

@author: joe
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd


#  准备数据
io = r'D:\My_article\第二篇\DATA\N-VIs.xlsx'
data_rf = pd.read_excel(io, sheet_name = 0)

x = np.array(data_rf.iloc[:,3:51])
y = np.array(data_rf.iloc[:,2])

# 随机森林回归
rfr = RandomForestRegressor()
# 训练
rfr.fit(x,y)

#CV=k_fold次数
scores = cross_val_score(rfr,x,y,cv=10)
scores_r2 = cross_val_score(rfr,x,y,cv=10,scoring='r2')
scores_RMSE = cross_val_score(rfr,x,y,cv=10,scoring='neg_mean_squared_error')
# 随机森林回归模型评估

print("随机森林回归的默认评估值为：", scores)
print("随机森林回归的R_squared值为：", scores_r2)
print("随机森林回归的均方误差为:", scores_RMSE)
print("随机森林回归的平均R_squared为:", np.mean(scores_r2))
print("随机森林回归的平均均方误差:", np.mean(scores_RMSE))



