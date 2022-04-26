from time import time
import re
import math
from itertools import *
import pandas as pd
from collections import Counter
from operator import itemgetter
import numpy as np
import scipy as sp


t = time()
data = [
    'a', 'b', 'is', 'python', 'jason', 'hello', 'hill', 'with', 'phone',
    'test', 'dfdf', 'apple', 'pddf', 'ind', 'basic', 'none', 'baecr', 'var',
    'bana', 'dd', 'wrd'
]



# dicr=[{dic[id]:({dp:dr})}  for (id,p,r) in users for (did,dp,dr) in users if id==did]
# print(dic)

# new_data=[]
# [new_data.append(u) for x in data if len(x)>3]
# print(new_data)
# words=["com.kjt.app/product_detail/101034","pre.kjt.com/product/101034","kjt.com/product_list/101034"
# ,"www.kjt.com/product/search/101034"]

# s=[x for x in words if len(re.findall(r"^\\(pre|search|list)",x))==0]
# print(s)

# df = pd.read_csv(name, encoding="utf-8")
# def set_data(data: list):
#     """
#         数据去重
#         :param data: 
#         :return: list
#         """
#     new_list = []
#     if data is None or len(data) == 0:
#         return new_list
#     # 先排序
#     data = [x for x in data if len(x) > 0]
#     sort_data = sorted(data, key=lambda x: x[1])
#     # 商品编号分组
#     group_data = groupby(data, key=lambda x: x[1])
#     indexes = []
#     for n in group_data:
#         indexes.append(str(n[0]))
#     # 合并商品编号重复的数据
#     for sysno in indexes:
#         count = 0.0
#         for d in sort_data:
#             if str(sysno) == str(d[1]):
#                 count += d[1]
#         new_list.append((data[0][0], sysno, count))
#     return new_list

# rst = "m.kjt.com/product/167244?from=singlemessage&isappinstalled=1"

# print(rst.replace("?from=singlemessage&isappinstalled=1", ""))
# reg = r"[0-9]+(?=[^0-9]*$)"
# rt = re.findall(
#     reg, "m.kjt.com/registertwo/sd?ReturnUrl=http://m.kjt.com/product/10002")

# # print(math.ceil((155142/1000)))

# lc1 = [(1001, 2), (1002, 3), (1003, 3)]

# for i,e in enumerate(lc1):
#     print(i,e)

# lc2 = [(1001, 0.1), (1003, 0.2), (1006, 0.3), (1005, 0.2)]

# lc1.extend(lc2)
# print(lc1)
# s=[(z,y,1)for  (z,y) in lc2]
# print(s)

# word=  123 
# print(word > 2147483647)

# friends = [(1001,"123",1,3),(1001,"100",1,2),(1001,"101",3,2),(1001,"102",5,3),(1001,"101",5,1)]

# df= pd.DataFrame(friends,index=None,columns=["userId", "productSysNo", "status", "qty"])
# df_visit=df.groupby(["productSysNo","status"],as_index=False).count()
# print(df_visit)
# result=[]
# for (productSysNo, customerSysNo,status,  qty) in df_visit.itertuples(index=False):
#     g_result= [ f for f in friends if productSysNo==f[1]]
#     print(g_result)
#     rq=0
#     for (u,p,s,q) in g_result:
#         rq+=q
#     result.append((g_result[0][0],g_result[0][1],rq))

# print(result)

things = [(1001, 10001, 1), (1003, 10001, 1), (1001, 10001, 1),
          (1002, 10001, 1), (1002, 10001, 1), (1004, 10001, 1)]
# # 必须要他么的先排序，不然分组不出来
# things=sorted(things,key=lambda x:x[0])  
# for key, items in groupby(things, itemgetter(0)):
#     print(key)
#     for subitem in items:
#         print(subitem)

# df = pd.DataFrame(thingsa, index=None, columns=["uid", "pid", "count"])
# df_group = df.groupby("pid", as_index=False)["uid", "count"].count()

# print(df_group)
somedata = [(1001, 10001, 2), (1001, 10002, 2), (1001, 10001, 2), (
    1001, 10003, 0.5), (1001, 10001, 2), (1001, 10004, 1), (1001, 10001, 2)]
somedata = sorted(somedata, key=lambda x: x[1])
# for key, items in groupby(somedata, lambda x: x[1]):
#     print(key)
#     rc=0
#     for (u,p,c) in items:
#         print((u,p,c))
#         rc+=c
#     print(somedata[0][0],key, rc)

# a=np.zeros((12,12),dtype='int32')
# print(a)
# print(a.shape[0])
# print(a.shape[1])
# print(a[-10:])

d = [[1, 2, 3,4,1,12], [4, 6,5, 6,2,23], [7, 9, 8,9,2,24], [7, 8, 9,9,4,32], [11, 23,12,13,5,32],
     [21,23, 14,6,1,15], [16,10, 24,36,8,54]]
d_matrix=np.array(d)
print(d_matrix)

# print(d_matrix.shape[0])  # 行
# print(d_matrix.shape[1])  # 列
# print("\n")

# for t in range(d_matrix.shape[0]):
#     print(sorted(d_matrix[t])[-4:])

# print("\n")
# print(d_matrix[0][-4:])         # 第1行后四个数字
# print(d_matrix[1][-4:])         # 第2行后四个数字
# print(d_matrix[2][-4:])         # 第3行后四个数字
# print(d_matrix[2][0:4])         # 第2行前四个数字
