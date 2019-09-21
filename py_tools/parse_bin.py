# coding=utf-8
import os
import re
import pandas as pd
from pandas.core.frame import DataFrame


def parse():
    df_data = pd.read_excel('bin.xlsx')
    # 默认读取前5行的数据
    data = df_data.head()
    print(data)
    svc_dict = list()
    for row in df_data.itertuples():
        name = getattr(row, '发卡行名称')
        length = getattr(row, '长度')
        val = getattr(row, '取值')
        c_type = getattr(row, '卡种')
        datepat = re.compile(r'\(.*?\)')
        b_name = re.sub(datepat, '1', name.replace('\n',''))
        body = (b_name.replace('1',''), length, val, c_type)
        svc_dict.append(body)

    df_list = DataFrame(svc_dict)
    df_list.columns = ['发卡行名称', '长度', '取值', '卡种']
    df_list.to_csv('bin.csv', encoding='utf_8_sig')

    print("finished")


if __name__ == "__main__":
    parse()
