import os
import pandas as pd
import xlwt

import xlrd
from xlutils.copy import copy
import sys

# 要写入的excel全路径及名称
file_to = sys.argv[1]
# 要合并的excel的目录
rootdir = sys.argv[2]
# 要合并的excel文件名的公共部分
file_name = sys.argv[3]


# 新建列表，存放文件名
filename_excel = []


# 列出文件夹下所有的目录与文件
def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


# 获取指定的文件名和要写入的sheet名
def get_file(files, file_name):
    # 存放该excel的list
    file_list = []
    # 存放文件去除后缀名的list
    file_name_list = []
    for file in files:
        # 判断此文件名称是否包含此关键名称
        if file_name in file:
            file_list.append(file)
            print(file)
            # 截取要写入的sheet名（此处截取的规则是：截取最后一个'/'之后，'.'之前，'）'之后）
            # /Users/ywq/Downloads/工作/温检/数据/市/2018年12月-2019年11月/050101_[1]合计_地区分布(行)_[27]结案、提出抗诉（件）（温州2018年12月-2019年11月）.xlsx
            # 结案、提出抗诉（件）（温州2018年12月-2019年11月）
            file_name_list.append(file.split('/')[-1].split('.')[0].split(')')[1])
    return file_list, file_name_list


def merge_files(file_to, files, file_names):
    # 创建sheet
    workbook = xlwt.Workbook()
    for file_name in file_names:
        workbook.add_sheet(file_name, cell_overwrite_ok=True)
        # print(file_name)
        # print(files[file_names.index(file_name)])
        workbook.save(file_to)

    # 向不同sheet写入数据
    writer = pd.ExcelWriter(file_to)
    for file_name in file_names:
        df = pd.read_excel(files[file_names.index(file_name)])
        # print(df.columns)
        # df.rename(columns={'Unnamed: 0': ''})
        df.to_excel(writer, sheet_name=file_name, startcol=0, index=False)
    writer.save()


    rb = xlrd.open_workbook(file_to, formatting_info=True)
    wb = copy(rb)
    # 修改第一个cell的值('Unnamed: 0' -> '')   这个步骤可根据实际情况选择
    for file_name in file_names:
        ws = wb.get_sheet(file_name)
        ws.write(0, 0, '')
    wb.save(file_to)

files, file_names = get_file(list_all_files(rootdir), file_name)
merge_files(file_to, files, file_names)