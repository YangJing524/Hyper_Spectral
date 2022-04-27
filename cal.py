import os
import sys

import xlwings as xw

wb = xw.Book("原始光谱.xlsx")

for s in range(6):      #worksheet
    sht = wb.sheets[s]
    sht.activate()
    rng = sht.range('A1').expand('table')   #确定当前worksheet有几行几列
    nrows = rng.rows.count
    ncols = rng.columns.count
    for x in range(1, ncols):
        for y in range(1, nrows):
            sht.range(y + 1, x + 1).formula = '=POWER(%s,-1)' % sht.range(y + 1, x + 1).value   #这个工具里xy是反过来的，所以y在前。把formula后面的内容改成对应的excel函数即可
            #print(x+1,y+1)
wb.save('倒数.xlsx')  #另存为
