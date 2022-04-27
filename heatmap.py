import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

plt.rc('font', family='Times New Roman')
plt.rcParams ['xtick.labelsize']=15
plt.rcParams ['ytick.labelsize']=15
plt.rcParams ['axes.titlesize']=15
plt.rcParams ['axes.labelsize']=15
from matplotlib import font_manager as fm
myfont=fm.FontProperties(fname="d:\Fonts\simkai.ttf")
# plt.xlable('x',fontproperties=myfont,fontsize=32)
cmap = sns.cm.rocket_r

path =r"D:\Data\Python\Hyper_Spectral\训练集\RMSE-R2图表.xlsx"

data1=pd.read_excel(path,sheet_name='R2')
_data1=np.array(data1)
data1[list(data1.columns)]=data1[list(data1.columns)].astype(float)

data2=pd.read_excel(path,sheet_name='RMSE')
data2[list(data2.columns)]=data2[list(data2.columns)].astype(float)
_data2=np.array(data2)
#子图设置
f,(ax1,ax2) = plt.subplots(figsize = (15, 5),ncols=2,sharey=True)
plt.subplots_adjust(top=0.9,bottom=0.1,left=0.05,right=1,wspace=0, hspace=0.2)
# cubehelix map颜色
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(_data1, linewidths = 0.5,linecolor='white',ax = ax1, annot=True,vmax=1, vmin=0.5,cmap='coolwarm',
            annot_kws={'size':12, 'color':'black'})
ax1.set_title('$R^{2}$')
ax1.set_xlabel('')
ax1.set_xticklabels(list(data1.columns)) #设置x轴图例为空值
ax1.set_yticklabels(list(data1.index))
label_y = ax1.get_yticklabels()
plt.setp(label_y , rotation = 360)

# matplotlib colormap
sns.heatmap(_data2, linewidths = 0.5, linecolor='white',annot=True, ax = ax2, vmax=5, vmin=2,cmap='coolwarm',
            annot_kws={'size':12, 'color':'black'})
# rainbow为 matplotlib 的colormap名称
ax2.set_title('RMSE')
ax2.set_xlabel('')
ax2.set_xticklabels(list(data2.columns)) #设置x轴图例为空值
ax2.set_yticklabels(list(data2.index))
label_y = ax2.get_yticklabels()
plt.setp(label_y , rotation = 360)

plt.savefig('R_2',dpi=300)
plt.show()

