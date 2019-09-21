# coding=utf-8
import os
import json
import requests
import pandas as pd
from pandas.core.frame import DataFrame

urls = [
    'http://ztcloudtest.jwell56.com/customer/v2/api-docs',
    'http://ztcloudtest.jwell56.com/product/v2/api-docs',
    'http://ztcloudtest.jwell56.com/order/v2/api-docs',
    'http://ztcloudtest.jwell56.com/pay/v2/api-docs',
    'http://ztcloudtest.jwell56.com/cms/v2/api-docs',
    'http://ztcloudtest.jwell56.com/auth/v2/api-docs',
    'http://ztcloudtest.jwell56.com/common/v2/api-docs',
    'http://ztcloudtest.jwell56.com/search/v2/api-docs',
    'http://ztcloudtest.jwell56.com/sso/v2/api-docs',
    'http://ztcloudtest.jwell56.com/aggregation/v2/api-docs',
    'http://ztcloudtest.jwell56.com/openapi/v2/api-docs',
    'http://ztcloudtest.jwell56.com/oms/v2/api-docs'
]
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36)"}
file_name = "scv.csv"


def parse():

    df_list = DataFrame()
    for url in urls:
        response = requests.get(url, headers=headers).text
        # 转化为字符串
        json_str = json.loads(response)
        # 大title
        title = json_str['info']['title']
        print(title)
        service_path = json_str['paths']
        svc_dict = list()
        for svc, data in service_path.items():
            req = data.get('post')
            req_method = 'post'
            if req == '' or req is None:
                req = data.get('get')
                req_method = 'get'
            if req == '' or req is None:
                req = data.get('put')
                req_method = 'put'
            if req == '' or req is None:
                req = data.get('delete')
                req_method = 'delete'
            if req is not None:
                body = (title, svc, req.get('summary'), req_method)
                svc_dict.append(body)

        if df_list.empty:
            df_list = DataFrame(svc_dict)
        else:
            df_list = df_list.append(DataFrame(svc_dict))

    df_list.columns = ['title', 'url', 'description', 'method']
    if os.path.exists(file_name):
        os.remove(file_name)
    df_list.to_csv('svc.csv', encoding='utf_8_sig')
    
    print("finished")


if __name__ == "__main__":
    parse()
