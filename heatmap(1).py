import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from matplotlib import font_manager as fm
myfont=fm.FontProperties(fname="d:\Fonts\simkai.ttf")
# plt.xlable('x',fontproperties=myfont,fontsize=32)
cmap = sns.cm.rocket_r

path =r"D:\Data\Python\Hyper_Spectral\训练集\RMSE图表.xlsx"

data1=pd.read_excel(path,sheet_name='R2')
_data1=np.array(data1)

data2=pd.read_excel(path,sheet_name='RMSE')
_data2=np.array(data2)

f, (ax1,ax2) = plt.subplots(figsize = (14, 6),ncols=2)
# cubehelix map颜色
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(_data1,ax = ax1, annot=True,vmax=1, vmin=0.5,cmap=cmap)
ax1.set_title('$R^{2}$')
ax1.set_xlabel('')
ax1.set_xticklabels(list(data1.columns)) #设置x轴图例为空值
ax1.set_yticklabels(list(data1.index))

# matplotlib colormap
sns.heatmap(_data2, linewidths = 0.05, annot=True, ax = ax2, vmax=12, vmin=0,cmap=cmap)
# rainbow为 matplotlib 的colormap名称
ax2.set_title('RMSE')
ax2.set_xticklabels(list(data2.columns)) #设置x轴图例为空值
ax2.set_yticklabels(list(data2.index))
plt.savefig('R_2',dpi=300)
plt.show()