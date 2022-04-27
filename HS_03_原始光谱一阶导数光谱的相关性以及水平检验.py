#coding=cp936
u'''
START:2019-02-06
AUTHOR:YangJing
USE:
    0.01相关性水平线
    0.05相关性水平线
OUTPUT:
       oringinal.png
       derivative.png
       corr_oringinal.txt
       corr_derivative.txt
       输出0.01以及0.05水平线与光谱的交点

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
import os
import argparse


parse = argparse.ArgumentParser ( description='add svc file and output the result' )
parse.add_argument ( '-svc', '--svc', help='the SVC EXCEL FILE' )
parse.add_argument ( '-spad', '--spad', help='the SPAD EXCEL FILE' )

args = parse.parse_args ()
svc_path = args.svc
spad_path = args.spad
full_path=os.path.split(svc_path)[0]

try:
    os.makedirs(os.path.join(full_path,'HS_03'))
except:
    pass
HS_path=os.path.join(full_path,'HS_03')

#求原始图像各波段相关系数与P值
def corr_p(data,spad,sheet):
    print('[INFO]处理原始光谱曲线')
    l1=[]
    l2=[]
    col=data.columns

    num=len(data.index)
    #start [index start 0]
    try:
        index=np.linspace(0,num-1,num)
        data.index=index
        spad.index=index
    except ValueError as e:
        print('spad长度与svc长度不相等：%s'%e)

    for i in col:
        #pearson函数返回两个值，分别为相关系数以及P值（显著性水平）
        #l1:相关系数列表，l2:p值列表
        value=pearsonr(spad[spad.columns[0]], data[i])
        l1.append(round(value[0],6))
        l2.append(round(value[1],3))
    corr_se=pd.Series(l1,index=col)
    
    p_se=pd.Series(l2,index=col)

    #因为不可避免的存在0.01,0.05水平线不存在，因此依次在附近寻找了+-0.003范围的值
    index_001_list=[0.010,0.011,0.009,0.012,0.008,0.007,0.013,0.006,0.014]
    index_005_list=[0.050,0.051,0.049,0.052,0.048,0.047,0.053,0.046,0.054]
    index_001=[]
    index_005=[]

    for i in index_001_list:
        index_001.append(list(p_se[p_se==i].index.values))
    for i in index_005_list:
        index_005.append(list(p_se[p_se==i].index.values))
    #数据清洗
    index_001_=[list(i) for i in index_001 if len(i)!=0]
    index_005_=[list(i) for i in index_005 if len(i)!=0]

    index_001=[i for j in index_001_ for i in j if corr_se[i]>0]
    index_001_01 = [i for j in index_001_ for i in j if corr_se[i]<0]

    index_005=[i for j in index_005_ for i in j if corr_se[i]>0]
    index_005_01 = [i for j in index_005_ for i in j if corr_se[i] < 0]

    #test
    print(index_001,index_005)
    print('#-'*20)
    print(index_001_01,index_005_01)

    ####20190207
    #p=0.01,p=0.05所对应波段的反射率
    try:
        p_001=corr_se[index_001[0]]
        p_001_01=corr_se[index_001_01[0]]
        p_005=corr_se[index_005[0]]
        p_005_01=corr_se[index_005_01[0]]
    except:
        pass
    def concat():
        se_1=pd.Series(p_001,index=col)
        se_2=pd.Series(p_001_01,index=col)
        se_3=pd.Series(p_005,index=col)
        se_4=pd.Series(p_005_01,index=col)
        se_all = pd.concat ( [corr_se, se_1, se_2,se_3,se_4], axis=1 )
        se_all.columns = ['R', 'p_001', 'p_001_01', 'p_005', 'p_005_01']
        return se_all

    #对单个值的横向填充为PD.SERIES



    idmax=corr_se.idxmax()
    idmin=corr_se.idxmin()
    print('p_001,p_005')
    #print(p_001,p_005)
    print('*')
    print('[INFO]idmax:%s,idmin:%s'%(idmax,idmin))
    #绘图
    s='oringinal_%s.png'%(sheet)
    xticks=np.arange(338,2538,200)
    draw(concat(),s,xticks,sheet)
    #写入txt文档，0.01,0.05交点用于分析
    with open(os.path.join(HS_path,'corr_original_%s.txt'%(sheet)),'w') as f:
        f.writelines(u'最大相关系数所对应波段:'+str(idmax)+'\n')
        f.writelines('相关系数最大值:'+str(corr_se[idmax])+'\n')
        f.writelines('负相关最大所对应波段:'+str(idmin)+'\n')
        f.writelines('负相关最大值:'+str(corr_se[idmin])+'\n')
        f.writelines('0.05水平线与正相关系数曲线交点:'+str(index_005)+'\n')
        f.writelines('0.05水平线与负相关系数曲线交点:'+str(index_005_01)+'\n')
        f.writelines('0.01水平线与正相关系数曲线交点:'+str(index_001)+'\n')
        f.writelines('0.01水平线与负相关系数曲线交点:'+str(index_001_01)+'\n')


def get_p_value(p_value,corr_se,name):
    #empty=np.zeros_like(corr_se)
    min_corr=corr_se.min()
    max_corr=corr_se.max()
    if min_corr*max_corr>0:
        se=pd.Series(p_value,index=corr_se.index)
        se.name=name
        return se
    else:
        #empty_1=empty.copy()
        #empty_1[:]=p_se
        se_1=pd.Series(p_value,index=corr_se.index)
        se_1.name=name
        #empty_2=empty.copy()
        #empty_2[:]=-p_se
        se_2=pd.Series(-p_value,index=corr_se.index)
        se_2.name=name+'_01'
        return pd.concat([se_1,se_2],axis=1)
def draw(corr,s,xticks):
    '''
    if corr.columns[0]==338:
        xticks=np.arange(338,2538,200)
        xlim=(338,2538)
    elif corr.columns[0]==339:
        xticks=np.arange(339,2539,200)
        xlim=(339,2539)
    UnboundLocalError:
    '''

    style={'R':'k','p_001':'k','p_001_01':'k','p_005':'k--','p_005_01':'k--'}
    corr.plot(style=style,xticks=xticks,xlim=(xticks.min(),xticks.max()),figsize=(12,9))
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.savefig(HS_path+'./'+s)

def diff_corr_p(data,spad,sheet):
    print('[INFO]处理一阶导数曲线')
    l1=[]
    l2=[]
    #相比原始图像，一阶导数要先转为np.array做差分处理，再转为dataframe
    array=np.array(data)
    diff_array=np.diff(array,axis=1)
    col = data.columns+1
    #339-2539
    col=col[:-1]
    num=len(data.index)
    index = np.linspace ( 0, num - 1, num )
    #start [index start 0] coordinate index
    try:
        data.index=index
        spad.index=index
    except ValueError as e:
        print('spad长度与svc长度不相等：%s'%e)

    #columns少1，因为差分处理
    data=pd.DataFrame(diff_array,columns=col,index=index)


    #输出Excel
    #data.to_excel('./output/diff.xlsx')
    for i in col:
        #pearsonr函数返回两个值，分别为相关系数以及P值（显著性水平）
        #l1:相关系数列表，l2:p值列表
        value=pearsonr(spad[spad.columns[0]],data[i])
        l1.append(value[0])
        l2.append(round(value[1],3))
    corr_se=pd.Series(l1,index=col)
    p_se=pd.Series(l2,index=col)

    #注意：！！！因为不可避免的存在0.01,0.05水平线不存在，因此依次在附近寻找了+-0.002范围的值
    index_001_list=[0.010,0.011,0.009,0.012,0.008,0.013,0.007]
    index_005_list=[0.050,0.051,0.049,0.052,0.048,0.053,0.047]
    index_001=[]
    index_001_01=[]
    index_005=[]
    index_005_01=[]
    for i in index_001_list:
        index_001.append(list(p_se[p_se==i].index.values))
    for i in index_005_list:
        index_005.append(list(p_se[p_se==i].index.values))
    #数据清洗
    index_001_=[list(i) for i in index_001 if len(list(i))!=0]
    index_005_=[list(i) for i in index_005 if len(list(i))!=0]
    mask_001=[corr_se[i]>=0 for i in index_001_]
    mask_005=[corr_se[i]>=0 for i in index_005_]

    index_001=[list(se[se==True].index.values) for se in mask_001]
    index_001_01=[list(se[se==False].index.values) for se in mask_001]

    index_005=[list(se[se==True].index.values) for se in mask_005]
    index_005_01=[list(se[se==False].index.values) for se in mask_005]

    print(index_001_,index_005_)
    #p=0.01,p=0.05所对应波段的derivative值
    p_001=corr_se[index_001[0][0]]
    p_001_01= corr_se[index_001_01[0][0]]

    p_005 = corr_se[index_005[0][0]]
    p_005_01= corr_se[index_005_01[0][0]]
    def concat():
        se_1=pd.Series(p_001,index=col)
        se_2=pd.Series(p_001_01,index=col)
        se_3=pd.Series(p_005,index=col)
        se_4=pd.Series(p_005_01,index=col)
        se_all = pd.concat ( [corr_se, se_1, se_2,se_3,se_4], axis=1 )
        se_all.columns=['R','p_001','p_001_01','p_005','p_005_01']
        return se_all
    #对单个值的横向填充为PD.SERIES
    #p_001_data=get_p_value(p_001,corr_se,name='p_001')
    #p_005_data=get_p_value(p_005,corr_se,name='p_005')
    #corr=pd.concat([corr_se,p_001_data,p_005_data],axis=1)

    idmax=corr_se.idxmax()
    idmin=corr_se.idxmin()
    print('p_001,p_005')
    print(p_001,p_005)
    print('*')
    print('[INFO]idmax:%s,idmin:%s'%(idmax,idmin))
    #绘图
    s='derivative_%s.png'%(sheet)
    xticks=np.arange(339,2539,200)
    draw(concat(),s,xticks)

    #写入txt文档，0.01,0.05交点用于分析
    with open(os.path.join(HS_path,'corr_diff_%s.txt'%(sheet)),'w') as f:
        f.writelines(u'最大相关系数所对应波段:'+str(idmax)+'\n')
        f.writelines('相关系数最大值:'+str(corr_se[idmax])+'\n')
        f.writelines('负相关最大所对应波段:'+str(idmin)+'\n')
        f.writelines('负相关最大值:'+str(corr_se[idmin])+'\n')
        f.writelines('0.05水平线与正相关系数曲线交点:'+str(index_005)+'\n')
        f.writelines('0.05水平线与负相关系数曲线交点:'+str(index_005_01)+'\n')
        f.writelines('0.01水平线与正相关系数曲线交点:'+str(index_001)+'\n')
        f.writelines('0.01水平线与负相关系数曲线交点:'+str(index_001_01)+'\n')
def main():
    starttime = datetime.datetime.now()
    print(__doc__)
    print('''该脚本可能会运行几秒钟，最终结果会保存在当前目录的F:/output/文件夹下，包括以下内容：
          1：输入：[svc.xlsx]
          2: 输入：所有小区spad值[spad.xlsx]
          3：输出：0.05水平，0.01水平的原始图像相关性检验[original.png]
          4：输出：0.05水平，0.01水平的一阶导数光谱相关性检验[derivative.png]
          5：输出：原始图像相关性最大波段的及相关系数[corr_original.txt]
          6：输出：一阶导数相关性最大波段的及相关系数[corr_diff.txt]
         说明：本人才疏学浅，对遥感反演原理不甚了解，数据处理中有诸多纰漏，望慎重使用，以免给各位带来不必要的麻烦。
         ''')

    print('...[INFO]加载数据集...')
    
    names=['0605','0622','0717','0826']
    for sheet in names:
        sig=pd.read_excel(svc_path,sheet_name=sheet)

        spad=pd.read_excel(spad_path,sheet_name=sheet,header=None)
        corr_p(sig.copy(),spad.copy(),sheet)
        diff_corr_p(sig.copy(),spad.copy(),sheet)
        

    endtime = datetime.datetime.now()
    print('-'*80)
    print('程序运行时间:%s s'%((endtime - starttime).seconds))


'''
corr_se[1334]
Out[28]: -0.16722162390032191

EMPTY_SE_001=np.zeros_like(corr_se)
EMPTY_SE_001[:]=-0.16722162390032191
se_01=pd.Series(EMPTY_SE_001,index=index)
corr=pd.concat([corr_se,se_01,se_05],axis=1)
'''
#diff_array=np.diff(sig_array,axis=1)
'''

diff_corr_se[diff_p_se[diff_p_se==0.01].index]
Out[100]:
991    -0.167330
1071   -0.167740
1100    0.166970
1215    0.166174
1232    0.167815
1308   -0.166201
1426   -0.166492
1709   -0.166323
1735   -0.167574
1819   -0.166347
2094   -0.167152
2129   -0.167569
2321    0.167649
2383    0.167009
2478    0.167431
2505    0.166817
dtype: float64
'''
'''
diff_1.idxmax()
Out[139]:
0    759
1    339
2    339
1    339
2    339
dtype: int64

diff_1[0][759]
Out[140]: 0.8388844431717504

diff_1.idxmin()
Out[141]:
0    523
1    339
2    339
1    339
2    339
dtype: int64

diff_1[0][523]
Out[142]: -0.7787709002181252
'''
main()