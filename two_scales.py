import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path=r'D:\Data\Python\Hyper_Spectral\训练集\比较RMSE.xlsx'

train = pd.read_excel(path,index_col=None)
test = pd.read_excel(path,index_col=None,sheet_name='Test')
plt.rc('font', family='Times New Roman')
plt.rcParams ['xtick.labelsize']=15
plt.rcParams ['ytick.labelsize']=15
plt.rcParams ['axes.titlesize']=15
plt.rcParams ['axes.labelsize']=15


# Create some mock data
X = list(train.columns)


fig, axes = plt.subplots(1,2,figsize=(12,5))

def plot_train(ax1=axes[0]):
    color = 'tab:red'
    ax1.set_xlabel('(a)')
    ax1.set_ylabel('$R^2$', color=color)
    # Index(['R2_M', 'R2_SVR', 'R2_RF', 'RMSE_M', 'RMSE_SVR', 'RMSE_RF'], dtype='object')
    ax1.plot(X, train.loc['R2_M'], color=color,linestyle='-',marker='*',label='$R^2\_ M$')
    ax1.plot(X, train.loc['R2_SVR'], color=color,linestyle='-',marker='^',label='$R^2\_ SVR$')
    ax1.plot(X, train.loc['R2_RF'], color=color,linestyle='-',marker='o',label='$R^2\_ RF$')
    ax1.tick_params(axis='y', labelcolor=color)
    h1=ax1.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0.)
    h1.get_frame().set_linewidth(0.0)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
    ax2.plot(X, train.loc['RMSE_M'], color=color,linestyle='--',marker='*',label='RMSE_M')
    ax2.plot(X, train.loc['RMSE_SVR'], color=color,linestyle='--',marker='^',label='RMSE_SVR')
    ax2.plot(X, train.loc['RMSE_RF'], color=color,linestyle='--',marker='o',label='RMSE_RF')
    ax2.tick_params(axis='y', labelcolor=color)
    h2=ax2.legend(bbox_to_anchor=(1.2, 0.8), loc='upper left', borderaxespad=0.)
    h2.get_frame().set_linewidth(0.0)
    ax1.grid(axis='y')

def plot_test(ax1=axes[1],train=test):
    color = 'tab:red'
    ax1.set_xlabel('(b)')
    ax1.set_ylabel('$R^2$', color=color)
    # Index(['R2_M', 'R2_SVR', 'R2_RF', 'RMSE_M', 'RMSE_SVR', 'RMSE_RF'], dtype='object')
    ax1.plot(X, train.loc['R2_M'], color=color,linestyle='-',marker='*',label='$R^2\_  M$')
    ax1.plot(X, train.loc['R2_SVR'], color=color,linestyle='-',marker='^',label='$R^2\_  SVR$')
    ax1.plot(X, train.loc['R2_RF'], color=color,linestyle='-',marker='o',label='$R^2\_  RF$')
    ax1.tick_params(axis='y', labelcolor=color)
    #h1=ax1.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0.)
    #h1.get_frame().set_linewidth(0.0)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
    ax2.plot(X, train.loc['RMSE_M'], color=color,linestyle='--',marker='*',label='RMSE_M')
    ax2.plot(X, train.loc['RMSE_SVR'], color=color,linestyle='--',marker='^',label='RMSE_SVR')
    ax2.plot(X, train.loc['RMSE_RF'], color=color,linestyle='--',marker='o',label='RMSE_RF')
    ax2.tick_params(axis='y', labelcolor=color)
    #h2=ax2.legend(bbox_to_anchor=(1.2, 0.8), loc='upper left', borderaxespad=0.)
    #h2.get_frame().set_linewidth(0.0)
    ax1.grid(axis='y')

plot_train()
plot_test()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('RMSE',dpi=600)

#plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

# import matplotlib
# matplotlib.axes.Axes.twinx
# matplotlib.axes.Axes.twiny
# matplotlib.axes.Axes.tick_params
