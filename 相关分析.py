from __future__ import print_function
import numpy as np
import pandas as pd

original = 'D:\My article\第二篇\DATA\相关性分析.xlsx' #餐饮数据，含有其他属性
data = pd.read_excel(original, sheet_name = 0)
x = np.array(data.iloc[:,2:])
y = np.array(data.iloc[:,1:])

#print(x)
#print("相关系数矩阵，即给出了任意两款菜式之间的相关系数:")
print(data.corr()) #相关系数矩阵，即给出了任意两款菜式之间的相关系数
print("显示“N”与其他菜式的相关系数:")
print(data.corr()[u'N']) #只显示“N”与其列的相关系数
print(max(data.corr()[u'N']))

#print(data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺'])) #计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数