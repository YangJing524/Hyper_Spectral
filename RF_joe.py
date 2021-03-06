# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:15:41 2020

@author: joe
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd


# 1 准备数据

io = r'D:\Data\Thesis\LCC-VIs.xlsx'
#io = r'C:\SIF\Multivariate1.xlsx'
data_rf = pd.read_excel(io, sheet_name = 0)
#data.head()

x = np.array(data_rf.iloc[:,1:])
#x = np.array(data_rf.iloc[:,22:29])

y = np.array(data_rf.iloc[:,0])#excel数据表中序列号的那个列不算下一列从0开始算列号

#boston = load_boston()

#x = boston.data
#y = boston.target

# 2 分割训练数据和测试数据
# 随机采样25%作为测试 75%作为训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 3 训练数据和测试数据进行标准化处理
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 4 三种集成回归模型进行训练和预测
# 随机森林回归
rfr = RandomForestRegressor()
# 训练
rfr.fit(x_train, y_train)
# 预测 保存预测结果
rfr_y_predict = rfr.predict(x_test)

# 5 模型评估
# 随机森林回归模型评估
print("随机森林回归的默认评估值为：", rfr.score(x_test, y_test))
print("随机森林回归的R_squared值为：", r2_score(y_test, rfr_y_predict))
print("随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                          ss_y.inverse_transform(rfr_y_predict)))
print("随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(rfr_y_predict)))
#print(x)
print(data_rf.iloc[:,0])


