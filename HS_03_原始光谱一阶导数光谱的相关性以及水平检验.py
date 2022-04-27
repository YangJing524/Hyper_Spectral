#coding=cp936
u'''
START:2019-02-06
AUTHOR:YangJing
USE:
    0.01�����ˮƽ��
    0.05�����ˮƽ��
OUTPUT:
       oringinal.png
       derivative.png
       corr_oringinal.txt
       corr_derivative.txt
       ���0.01�Լ�0.05ˮƽ������׵Ľ���

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

#��ԭʼͼ����������ϵ����Pֵ
def corr_p(data,spad,sheet):
    print('[INFO]����ԭʼ��������')
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
        print('spad������svc���Ȳ���ȣ�%s'%e)

    for i in col:
        #pearson������������ֵ���ֱ�Ϊ���ϵ���Լ�Pֵ��������ˮƽ��
        #l1:���ϵ���б�l2:pֵ�б�
        value=pearsonr(spad[spad.columns[0]], data[i])
        l1.append(round(value[0],6))
        l2.append(round(value[1],3))
    corr_se=pd.Series(l1,index=col)
    
    p_se=pd.Series(l2,index=col)

    #��Ϊ���ɱ���Ĵ���0.01,0.05ˮƽ�߲����ڣ���������ڸ���Ѱ����+-0.003��Χ��ֵ
    index_001_list=[0.010,0.011,0.009,0.012,0.008,0.007,0.013,0.006,0.014]
    index_005_list=[0.050,0.051,0.049,0.052,0.048,0.047,0.053,0.046,0.054]
    index_001=[]
    index_005=[]

    for i in index_001_list:
        index_001.append(list(p_se[p_se==i].index.values))
    for i in index_005_list:
        index_005.append(list(p_se[p_se==i].index.values))
    #������ϴ
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
    #p=0.01,p=0.05����Ӧ���εķ�����
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

    #�Ե���ֵ�ĺ������ΪPD.SERIES



    idmax=corr_se.idxmax()
    idmin=corr_se.idxmin()
    print('p_001,p_005')
    #print(p_001,p_005)
    print('*')
    print('[INFO]idmax:%s,idmin:%s'%(idmax,idmin))
    #��ͼ
    s='oringinal_%s.png'%(sheet)
    xticks=np.arange(338,2538,200)
    draw(concat(),s,xticks,sheet)
    #д��txt�ĵ���0.01,0.05�������ڷ���
    with open(os.path.join(HS_path,'corr_original_%s.txt'%(sheet)),'w') as f:
        f.writelines(u'������ϵ������Ӧ����:'+str(idmax)+'\n')
        f.writelines('���ϵ�����ֵ:'+str(corr_se[idmax])+'\n')
        f.writelines('������������Ӧ����:'+str(idmin)+'\n')
        f.writelines('��������ֵ:'+str(corr_se[idmin])+'\n')
        f.writelines('0.05ˮƽ���������ϵ�����߽���:'+str(index_005)+'\n')
        f.writelines('0.05ˮƽ���븺���ϵ�����߽���:'+str(index_005_01)+'\n')
        f.writelines('0.01ˮƽ���������ϵ�����߽���:'+str(index_001)+'\n')
        f.writelines('0.01ˮƽ���븺���ϵ�����߽���:'+str(index_001_01)+'\n')


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
    print('[INFO]����һ�׵�������')
    l1=[]
    l2=[]
    #���ԭʼͼ��һ�׵���Ҫ��תΪnp.array����ִ�����תΪdataframe
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
        print('spad������svc���Ȳ���ȣ�%s'%e)

    #columns��1����Ϊ��ִ���
    data=pd.DataFrame(diff_array,columns=col,index=index)


    #���Excel
    #data.to_excel('./output/diff.xlsx')
    for i in col:
        #pearsonr������������ֵ���ֱ�Ϊ���ϵ���Լ�Pֵ��������ˮƽ��
        #l1:���ϵ���б�l2:pֵ�б�
        value=pearsonr(spad[spad.columns[0]],data[i])
        l1.append(value[0])
        l2.append(round(value[1],3))
    corr_se=pd.Series(l1,index=col)
    p_se=pd.Series(l2,index=col)

    #ע�⣺��������Ϊ���ɱ���Ĵ���0.01,0.05ˮƽ�߲����ڣ���������ڸ���Ѱ����+-0.002��Χ��ֵ
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
    #������ϴ
    index_001_=[list(i) for i in index_001 if len(list(i))!=0]
    index_005_=[list(i) for i in index_005 if len(list(i))!=0]
    mask_001=[corr_se[i]>=0 for i in index_001_]
    mask_005=[corr_se[i]>=0 for i in index_005_]

    index_001=[list(se[se==True].index.values) for se in mask_001]
    index_001_01=[list(se[se==False].index.values) for se in mask_001]

    index_005=[list(se[se==True].index.values) for se in mask_005]
    index_005_01=[list(se[se==False].index.values) for se in mask_005]

    print(index_001_,index_005_)
    #p=0.01,p=0.05����Ӧ���ε�derivativeֵ
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
    #�Ե���ֵ�ĺ������ΪPD.SERIES
    #p_001_data=get_p_value(p_001,corr_se,name='p_001')
    #p_005_data=get_p_value(p_005,corr_se,name='p_005')
    #corr=pd.concat([corr_se,p_001_data,p_005_data],axis=1)

    idmax=corr_se.idxmax()
    idmin=corr_se.idxmin()
    print('p_001,p_005')
    print(p_001,p_005)
    print('*')
    print('[INFO]idmax:%s,idmin:%s'%(idmax,idmin))
    #��ͼ
    s='derivative_%s.png'%(sheet)
    xticks=np.arange(339,2539,200)
    draw(concat(),s,xticks)

    #д��txt�ĵ���0.01,0.05�������ڷ���
    with open(os.path.join(HS_path,'corr_diff_%s.txt'%(sheet)),'w') as f:
        f.writelines(u'������ϵ������Ӧ����:'+str(idmax)+'\n')
        f.writelines('���ϵ�����ֵ:'+str(corr_se[idmax])+'\n')
        f.writelines('������������Ӧ����:'+str(idmin)+'\n')
        f.writelines('��������ֵ:'+str(corr_se[idmin])+'\n')
        f.writelines('0.05ˮƽ���������ϵ�����߽���:'+str(index_005)+'\n')
        f.writelines('0.05ˮƽ���븺���ϵ�����߽���:'+str(index_005_01)+'\n')
        f.writelines('0.01ˮƽ���������ϵ�����߽���:'+str(index_001)+'\n')
        f.writelines('0.01ˮƽ���븺���ϵ�����߽���:'+str(index_001_01)+'\n')
def main():
    starttime = datetime.datetime.now()
    print(__doc__)
    print('''�ýű����ܻ����м����ӣ����ս���ᱣ���ڵ�ǰĿ¼��F:/output/�ļ����£������������ݣ�
          1�����룺[svc.xlsx]
          2: ���룺����С��spadֵ[spad.xlsx]
          3�������0.05ˮƽ��0.01ˮƽ��ԭʼͼ������Լ���[original.png]
          4�������0.05ˮƽ��0.01ˮƽ��һ�׵�����������Լ���[derivative.png]
          5�������ԭʼͼ���������󲨶εļ����ϵ��[corr_original.txt]
          6�������һ�׵����������󲨶εļ����ϵ��[corr_diff.txt]
         ˵�������˲���ѧǳ����ң�з���ԭ�����˽⣬���ݴ�����������©��������ʹ�ã��������λ��������Ҫ���鷳��
         ''')

    print('...[INFO]�������ݼ�...')
    
    names=['0605','0622','0717','0826']
    for sheet in names:
        sig=pd.read_excel(svc_path,sheet_name=sheet)

        spad=pd.read_excel(spad_path,sheet_name=sheet,header=None)
        corr_p(sig.copy(),spad.copy(),sheet)
        diff_corr_p(sig.copy(),spad.copy(),sheet)
        

    endtime = datetime.datetime.now()
    print('-'*80)
    print('��������ʱ��:%s s'%((endtime - starttime).seconds))


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