import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sheet=['','',]
for i in sheet:
    train = pd.read_excel('D:\Data\Python\Hyper_Spectral\训练集\SPAD原始.xlsx',sheetname=i)
    axes=plt.subplot(2,3,figsize=(15,9))
    for ax in axes:
        ax.boxplot(x=train,
                    patch_artist=None,
                    boxprops={'color':'blue'},
                    whiskerprops = {'color': "black",},
                    capprops = {'color': "black"},
                    flierprops={'color': 'red'},
                    medianprops={'color':'red'},)








