'''
START:2019-01-30
AUTHOR:YangJing
USE:SVC batch processing
'''

import numpy as np
import pandas as pd
import time
import os
from tqdm import trange
import argparse

print(__doc__)
t=time.ctime()
print('Current time:%s'%time.ctime())
print('-'*80)


def extract_sig(sig_path):

    f=open(sig_path,'r')
    data=pd.read_table(f,skiprows=25, sep=' ',header=None) #skiprows=4是头文件的行数
    data.columns=list('ABCDEFG')

    new = pd.DataFrame ( data['G'].values, index=data['A'].values, columns=['sig'] )
    #print(new)
    new = new/100
    name = sig_path[:-4]
    name=os.path.split(name)[-1]

    l = list ( new['sig'].values )
    #sig_series=pd.Series(l,index=index)
    return name,l


if __name__=='__main__':
    parse=argparse.ArgumentParser(description='add svc file and output the result')
    parse.add_argument('-f','--folder',help='the SVC folder')
    parse.add_argument ( '-date', '--date', help='the SVC folder' )


    args=parse.parse_args()
    path=args.folder
    date=args.date


    files=os.listdir(path)
    last_name=os.path.splitext(path)[-1]
    d={}
    print('START RUNNING')

    for i in trange(len(files)):
        sig=extract_sig(os.path.join(path,files[i]))
        d[sig[0]]=sig[1]
    index=np.arange(350,2501,1) #这里的350和2501是整个波段范围，1是重采样的间隔
    sig_data=pd.DataFrame(d,index=index)

    if os.path.exists(r'C:\Users\TINA\Desktop/output/'+date) == True:
        pass
    else:
        os.makedirs(r'C:\Users\TINA\Desktop/output/'+date)
    sig_data.T.to_excel(r'C:\Users\TINA\Desktop/output/'+date+'/'+r'%s.xlsx'%date)
    print(r'处理完成，处理结果见：C:\Users\TINA\Desktop/output/')