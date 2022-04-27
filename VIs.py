import math

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
import os
import argparse
import xlrd
import time


starttime = datetime.datetime.now()
print('START RUNNING...')
print('Current time:%s' % time.ctime())
print('-' * 80)
parse = argparse.ArgumentParser(description='add svc file and spad file')
parse.add_argument('-svc', '--svc', help='the SVC  FILE')
args = parse.parse_args()
full_path = args.svc
HS_path = os.path.split(full_path)[0]
print('若长度不相等，按 Ctrl+C 终止')



# 1：NDVI
def I_1(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][670]) / (data.iloc[i][800] + data.iloc[i][670]) for i in range(LENGTH)]
    return l
# 2：GNDVI
def I_2(data, LENGTH):
    l = [(data.iloc[i][801] - data.iloc[i][550])/(data.iloc[i][801] + data.iloc[i][550]) for i in range(LENGTH)]
    return l
#3. DVI
def I_3(data, LENGTH):
    l =[data.iloc[i][765]-data.iloc[i][719] for i in range(LENGTH)]
    return l
#4. RVI
def I_4(data, LENGTH):
    l = [data.iloc[i][766]/data.iloc[i][720] for i in range(LENGTH)]
    return l
#5. SAVI
def I_5(data, LENGTH):
    l = [1.5*(data.iloc[i][800]-data.iloc[i][670])/(data.iloc[i][800]+data.iloc[i][670]+0.5) for i in range(LENGTH)]
    return l
#6. CARI
def I_6(data, LENGTH):
    l = [(data.iloc[i][700]-data.iloc[i][670])/0.2*(data.iloc[i][800]+data.iloc[i][670]) for i in range(LENGTH)]
    return l
#7. TCARI
def I_7(data, LENGTH):
    l = [3*((data.iloc[i][700]-data.iloc[i][670])-0.2*(data.iloc[i][700]-data.iloc[i][500])*(data.iloc[i][700]/data.iloc[i][670]))
         for i in range(LENGTH)]
    return l
#8. MCARI
def I_8(data, LENGTH):
    l = [((data.iloc[i][700]-data.iloc[i][670])-0.2*(data.iloc[i][700]-data.iloc[i][550]))
           *(data.iloc[i][700]/data.iloc[i][670]) for i in range(LENGTH)]
    return l
#9. HNDVI
def I_9(data, LENGTH):
    l = [(data.iloc[i][827]-data.iloc[i][668])/(data.iloc[i][827]+data.iloc[i][668]) for i in range(LENGTH)]
    return l
#10. OSAVI
def I_10(data, LENGTH):
    l = [1.16*(data.iloc[i][800]-data.iloc[i][670])/(data.iloc[i][800]-data.iloc[i][670]+0.16) for i in range(LENGTH)]
    return l
#11. MTCI
def I_11(data, LENGTH):
    l = [((data.iloc[i][754])-(data.iloc[i][709]))/((data.iloc[i][709])-(data.iloc[i][684])) for i in range(LENGTH)]
    return l
#12 PRI
def I_12(data, LENGTH):
    l = [(data.iloc[i][570] - data.iloc[i][531]) / (data.iloc[i][570] + data.iloc[i][531]) for i in range(LENGTH)]
    return l
#13. SIPI
def I_13(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][450]) / (data.iloc[i][800] + data.iloc[i][450]) for i in range(LENGTH)]
    return l
#14.PSNDa
def I_14(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][680]) / (data.iloc[i][800]+data.iloc[i][680]) for i in range(LENGTH)]
    return l
#15 PSNDb
def I_15(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][635])/(data.iloc[i][800]+data.iloc[i][635]) for i in range(LENGTH)]
    return l
#16.PSSRa
def I_16(data, LENGTH):
    l = [data.iloc[i][800]/data.iloc[i][680] for i in range(LENGTH)]
    return l
#17. PSSRb
def I_17(data, LENGTH):
    l = [data.iloc[i][800]/data.iloc[i][635] for i in range(LENGTH)]
    return l
#18. CIred edge
def I_18(data, LENGTH):
    l = [(data.iloc[i][800]/data.iloc[i][670])-1 for i in range(LENGTH)]
    return l
#19. VARIred
def I_19(data, LENGTH):
    l = [(data.iloc[i][700]-1.7*data.iloc[i][670]+0.7*data.iloc[i][450])/(data.iloc[i][700]+2.3*data.iloc[i][670]-1.3*data.iloc[i][450])
         for i in range(LENGTH)]
    return l
#20.SR
def I_20(data, LENGTH):
    l = [(data.iloc[i][744]) / (data.iloc[i][667]) for i in range(LENGTH)]
    return l
#21. TVI
def I_21(data, LENGTH):
    l = [60*(data.iloc[i][800]-data.iloc[i][550])/100*(data.iloc[i][670]-data.iloc[i][550]) for i in range(LENGTH)]
    return l
#22. VOG1
def I_22(data, LENGTH):
    l = [(data.iloc[i][740]) / (data.iloc[i][720]) for i in range(LENGTH)]
    return l
#23.VOG2
def I_23(data, LENGTH):
    l = [(data.iloc[i][734] - data.iloc[i][747]) / (data.iloc[i][715]+data.iloc[i][726]) for i in range(LENGTH)]
    return l
#24.VOG3
def I_24(data, LENGTH):
    l = [(data.iloc[i][734] - data.iloc[i][747]) / (data.iloc[i][715]+data.iloc[i][720]) for i in range(LENGTH)]
    return l
#25.MRESR
def I_25(data, LENGTH):
    l = [(data.iloc[i][750] - data.iloc[i][445]) / (data.iloc[i][705]+data.iloc[i][445]) for i in range(LENGTH)]
    return l
#26.ARVI
def I_26(data, LENGTH):
    l = [(data.iloc[i][800] - data.iloc[i][670]) / (data.iloc[i][800]+data.iloc[i][670]-data.iloc[i][450]) for i in range(LENGTH)]
    return l
#27. NPCI
def I_27(data, LENGTH):
    l = [(data.iloc[i][680] - data.iloc[i][430]) / (data.iloc[i][680] + data.iloc[i][430]) for i in range(LENGTH)]
    return l
#28. GRVI
def I_28(data, LENGTH):
    l = [data.iloc[i][800]/data.iloc[i][550] for i in range(LENGTH)]
    return l
#29.RNDVI
def I_29(data, LENGTH):
    l = [(data.iloc[i][800]-data.iloc[i][670])/ np.sqrt(data.iloc[i][800]+data.iloc[i][670])for i in range(LENGTH)]
    return l
#30.MSR
def I_30(data, LENGTH):
    l = [(data.iloc[i][800]/data.iloc[i][670]-1) / (np.sqrt(data.iloc[i][800]/data.iloc[i][670])+1) for i in
         range(LENGTH)]
    return l
#31.NPQI
def I_31(data, LENGTH):
    l = [(data.iloc[i][415] - data.iloc[i][435]) / (data.iloc[i][415] + data.iloc[i][435]) for i in range(LENGTH)]
    return l
#32. IPVI
def I_32(data, LENGTH):
    l = [data.iloc[i][800] / (data.iloc[i][800] + data.iloc[i][670]) for i in range(LENGTH)]
    return l
#33.TVI1
def I_33(data, LENGTH):
    l = [(0.6*(data.iloc[i][800] - data.iloc[i][550])-(data.iloc[i][670]-data.iloc[i][550])) for i in range(LENGTH)]
    return l
#34.TVI2
def I_34(data, LENGTH):
    l = [(60*(data.iloc[i][800]-data.iloc[i][550])-100*(data.iloc[i][670]-data.iloc[i][550])) for i in range(LENGTH)]
    return l
#35.MTVI
def I_35(data, LENGTH):
    l = [(1.2*(1.2*(data.iloc[i][800]-data.iloc[i][550])-2.5*(data.iloc[i][670]-data.iloc[i][550]))) for i in range(LENGTH)]
    return l
#36. MTVI2
#def I_36(data,LENGTH):
   # l=[(1.5*(1.2*(data.iloc[i][800]-data.iloc[i][550])-2.5*(data.iloc[i][670]-data.iloc[i][550]))/
     #  np.sqrt((2*data.iloc[i][800]+1)*(2*data.iloc[i][800]+1)-(6*data.iloc[i][800]-5*np.sqrt(data.iloc[i][670]))-0.5)) for i in range(LENGTH)]
    #return l
#37. RENDVI
#def I_37(data, LENGTH):
   # l = [((data.iloc[i][750] - data.iloc[i][705])/(data.iloc[i][750] + data.iloc[i][705])) for i in range(LENGTH)]
   # return l
#38. MRENDVI
#def I_38(data, LENGTH):
    #l = [((data.iloc[i][750] - data.iloc[i][705])/(data.iloc[i][750] + data.iloc[i][705]-2*data.iloc[i][445])) for i in range(LENGTH)]
    #return l
#39.PSRI
#def I_39(data, LENGTH):
    #l = [(data.iloc[i][680] - data.iloc[i][550])/(data.iloc[i][750]) for i in range(LENGTH)]
    #return l
#40. NDNI
#def I_40(data, LENGTH):
    #l = [((math.log(1/data.iloc[i][1510])-math.log(1/data.iloc[i][1680]))/
        # (math.log(1/data.iloc[i][1510]) + math.log(1/data.iloc[i][1680]))) for i in range(LENGTH)]
    #return l
#41 MSI
def I_41(data, LENGTH):
    l = [(data.iloc[i][1599]) / (data.iloc[i][819]) for i in range(LENGTH)]
    return l
#42 NDII
def I_42(data, LENGTH):
    l = [(data.iloc[i][819] - data.iloc[i][1649]) / (data.iloc[i][819] + data.iloc[i][1649]) for i in range(LENGTH)]
    return l
#43. NDWI
def I_43(data, LENGTH):
    l = [(data.iloc[i][857] - data.iloc[i][1241]) / (data.iloc[i][857] + data.iloc[i][1241]) for i in range(LENGTH)]
    return l
#44.NMDI
def I_44(data, LENGTH):
    l = [((data.iloc[i][860] - (data.iloc[i][1640]-data.iloc[i][2130])) /
         (data.iloc[i][860]+(data.iloc[i][1640]-data.iloc[i][2130]))) for i in range(LENGTH)]
    return l
#45. WBI
def I_45(data, LENGTH):
    l = [(data.iloc[i][970]) / (data.iloc[i][900]) for i in range(LENGTH)]
    return l
# 46: mSR
def I_46(data, LENGTH):
    l = [(data.iloc[i][750] - data.iloc[i][446]) / (data.iloc[i][706] + data.iloc[i][446]) for i in range(LENGTH)]
    return l
# 47: PPR
def I_47(data, LENGTH):
    l = [(data.iloc[i][504] - data.iloc[i][436]) / (data.iloc[i][504] + data.iloc[i][436]) for i in range(LENGTH)]
    return l
#48 NDSI
def I_48(data, LENGTH):
    l = [(data.iloc[i][814] - data.iloc[i][762]) / (data.iloc[i][814] + data.iloc[i][762]) for i in range(LENGTH)]
    return l
# 49 LCI
def I_49(data, LENGTH):
    l = [(data.iloc[i][850] - data.iloc[i][710]) / (data.iloc[i][850] - data.iloc[i][680]) for i in range(LENGTH)]
    return l
#50. EVI
def I_50(data, LENGTH):
    l = [(2.5*(data.iloc[i][800] - data.iloc[i][670])/
          (data.iloc[i][800]+6*(data.iloc[i][760])-7.5*(data.iloc[i][450])+1)) for i in range(LENGTH)]
    return l

# 以上部分写入植被指数的计算式

def corr():
    print('[INFO]正在处理，请稍后...')
    workbook = xlrd.open_workbook(full_path)
    names = workbook.sheet_names()

    for period in names:
        print('正在处理第 %s 期' % period)
        # water=pd.read_excel(water_path,sheet_name=period,header=None)
        data = pd.read_excel(full_path, sheet_name=period)
        LENGTH = len(data.index)
        d = {'NDVI': I_1(data, LENGTH), 'GNDVI': I_2(data, LENGTH), 'DVI': I_3(data, LENGTH), 'RVI': I_4(data, LENGTH),'SAVI': I_5(data, LENGTH),
             'CARI':I_6(data,LENGTH),'TCARI':I_7(data,LENGTH),'MCARI':I_8(data,LENGTH),'HNDVI': I_9(data, LENGTH),'OSAVI':I_10(data,LENGTH),
             'MTCI': I_11(data, LENGTH), 'PRI': I_12(data, LENGTH),'SIPI': I_13(data, LENGTH), 'PSNDa': I_14(data, LENGTH),'PSNDb': I_15(data, LENGTH),
             'PSSRa': I_16(data, LENGTH),'PSSRb': I_17(data, LENGTH),'CIred edge': I_18(data, LENGTH),'VARIred': I_19(data, LENGTH), 'SR': I_20(data, LENGTH),
             'TVI': I_21(data, LENGTH),'VOG1': I_22(data, LENGTH), 'VOG2': I_23(data, LENGTH),'VOG3': I_24(data, LENGTH), 'MRESR': I_25(data, LENGTH),
             'ARVI': I_26(data, LENGTH),'NPCI': I_27(data, LENGTH), 'GRVI': I_28(data, LENGTH),'RNDVI': I_29(data, LENGTH), 'MSR': I_30(data, LENGTH),
             'NPQI': I_31(data, LENGTH),'IPVI': I_32(data, LENGTH), 'TVI1': I_33(data, LENGTH),'TVI2':I_34(data, LENGTH),'MTVI':I_35(data, LENGTH),
             #'MTVI2':I_36(data, LENGTH),'RENDVI': I_37(data, LENGTH), 'MRENDVI': I_38(data, LENGTH),'PSRI': I_39(data, LENGTH), 'NDNI': I_40(data, LENGTH),
             'MSI': I_41(data, LENGTH),'NDII': I_42(data, LENGTH), 'NDWI': I_43(data, LENGTH),'NMDI': I_44(data, LENGTH), 'WBI': I_45(data, LENGTH),
             'mSR': I_46(data, LENGTH),'PPR': I_47(data, LENGTH),'NDSI': I_48(data, LENGTH),'LCI': I_49(data, LENGTH),'EVI':I_50(data, LENGTH) }
# 这里按照上面的格式写入输出的植被指数名字
        index = data.index
        INDEX = pd.DataFrame(d, index=index)
        col = INDEX.columns

        # INDEX_1=pd.concat([water,INDEX],axis=1)
        INDEX.to_excel(os.path.join(HS_path, '%s-VIs.xlsx' % period))


def main():
    print('[INFO]加载数据集...')
    corr()
    endtime = datetime.datetime.now()
    print('-' * 60)
    print('程序运行时间:%s s' % ((endtime - starttime).seconds))


main()
