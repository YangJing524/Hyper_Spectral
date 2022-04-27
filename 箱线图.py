import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xlrd
import os

import time

plt.style.use('ggplot')

plt.rcParams ['axes.unicode_minus']=False
plt.rcParams ['font.sans-serif']=['SimHei']
plt.rc('font', family='Times New Roman')

sns.set_style("whitegrid")  #设置背景网格
SPAD = pd.read_excel('D:\Data\Python\Hyper_Spectral\训练集\SPAD原始.xlsx')
ax=sns.boxplot(data=SPAD,
            patch_artist=None,


            boxprops={'color':'blue'},

            whiskerprops = {'color': "black",},
            capprops = {'color': "black"},
            flierprops={'color': 'red'},
            medianprops={'color':'red'},
            )  #中位线设置为红色





plt.ylim(0,80)
plt.xlabel('Growth  Stages')
plt.ylabel('LCC')
#ax.set_xlim(0,4)
ax.set_ylim(20,70)
plt.savefig('LCC',dpi=300)

print('FINISH.')

