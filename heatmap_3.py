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

path =r"D:\Data\Python\Hyper_Spectral\训练集\相关分析.xlsx"

data1=pd.read_excel(path,sheet_name='ST')
_data1=np.array(data1)
#data1[list(data1.columns)]=data1[list(data1.columns)].astype(float)

data2=pd.read_excel(path,sheet_name='VI')
#data2[list(data2.columns)]=data2[list(data2.columns)].astype(float)
_data2=np.array(data2)

data3=pd.read_excel(path,sheet_name='TD')
#data3[list(data3.colums)]=data3[list(data3.colums)].astype(float)
_data3=np.array(data3)

grid_kws = {"height_ratios": (.9, .05), "hspace": .3}


f,(ax1,ax2,ax3,cbar_ax) = plt.subplots(ncols=1,nrows=3,sharex=True,gridspec_kw=grid_kws)
# cubehelix map颜色
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(_data1, linewidths = 0.05,ax = ax1, annot=True,vmax=1, vmin=-1,cmap=cmap,center=0)
ax1.set_title('ST')
ax1.set_xlabel('')
ax1.set_xticklabels(list(data1.columns)) #设置x轴图例为空值
ax1.set_yticklabels(list(data1.index))

# matplotlib colormap
sns.heatmap(_data2, linewidths = 0.05, annot=True, ax = ax2, vmax=1, vmin=-1,cmap=cmap,center=0)
# rainbow为 matplotlib 的colormap名称
ax2.set_title('VI')
ax2.set_xticklabels(list(data2.columns)) #设置x轴图例为空值
ax2.set_yticklabels(list(data2.index))

sns.heatmap(_data3, linewidths = 0.05,ax = ax3, annot=True,vmax=1, vmin=-1,cmap=cmap,center=0, cbar_ax=cbar_ax,cbar_kws={"orientation": "horizontal"})
ax3.set_title('TD')

ax3.set_xticklabels(list(data3.columns)) #设置x轴图例为空值
ax3.set_yticklabels(list(data3.index))


plt.show()