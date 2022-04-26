import numpy as np
from scipy import sparse
import numpy as np
import re
import time
import os
from itertools import *

def matrix_main():
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    # todense 转换稠密矩阵
    c = sparse.csc_matrix(
        (data, indices, indptr), shape=(3, 3)).todense()  # 按col列压缩
    r = sparse.csr_matrix(
        (data, indices, indptr), shape=(3, 3)).todense()  # 按row行压缩

    # print(indices[0:1])
    # print(indices[1:2])
    # print(indices[2:3])
    # print(indices[3:4])
    # print(indices[4:5])
    # print(indices[5:6])

    # print(c)
    # print(r)
    # list1 = [[1, 2, 3,4], [4, 5, 6,6], [7, 8, 9,9], [7, 8, 9,9]]
    # M = mat(list1.copy())

    ar = np.arange(12)
    ar.shape = (3, 4)
    # 生成矩阵
    m = np.mat(ar.copy())

    # #print(m[0:1])
    # #print(m[1:2])
    # #print(m[2:3])
    # #print(m[3:4])

    # print(m[:, :])

    # d = [[1, 2, 3,4], [4, 5, 6,6], [7, 8, 9,9], [7, 8, 9,9]]
    # d = [[200001, 300001,41], [400001, 50001, 611], [711022, 800001,90], [700002, 800002, 9]]
    # dmatrix=np.array(d)
    # print(dmatrix.shape[1])
    # x_p =dmatrix[:, :2] 
    # y=(sparse.csc_matrix((dmatrix[:, 2], x_p.T),dtype=np.int64))[:, :].toarray()
    # print(y)

    # dm=np.array(d)

    # tlist=[(1002,"product/102"),(1002,"product/102"),(1002,"prekjt.com/product/102"),(1002,"kjt.com/product/105"),
    # (1002,"kjt.com/product/arrivenotice/102"),(1002,"kjt.com/product/102"),(1002,"kjt.com/product/103"),
    # (1002,"kjt.com/product/searchresult/102"),(1002,"kjt.com.pre/product/102"),(1002,"kjt.com/searchinstore/102"),
    # (1002,"kjt.com.pre/product/1018")]

    # noprelist=[(id,url) for (id,url) in tlist if len(re.findall("pre",url))==0 and len(re.findall("arrivenotice",url))==0 
    # and len(re.findall("search",url))==0]


    myList = [('1001', 1), ('1001',2), ('1002',37), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4)
    , ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4)
    , ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4), ('1003', 4)]
    
    threads_num=5
    per_thread = int(len(myList) /threads_num) 
    # print(per_thread)
    # print(len(myList))
    sdata=[myList[i:i+per_thread] for i in range(0,len(myList),per_thread)]
  
    
    # for i in range(threads_num):

    #     if threads_num -i ==1:
    #         print("last")
    #     else:
    #       print(myList[i*per_thread:per_thread])  
    
    # group_data = groupby(myList, key=lambda x: x[0])
    # indexs = []
    # for n in group_data:
    #     indexs.append(str(n[0]))
    # print(indexs)

    user_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    threads_num = 7 
    per_thread = int(len(user_list) / threads_num)

    for i in range(threads_num):
        if threads_num-i==1:
            print(user_list[i*per_thread:])
        else:
            print(user_list[i*per_thread:per_thread*(i+1)])


def slic():
    arr= np.arange(12).reshape([3,4])
    print(arr,end="\n\n")
    print(arr[0::2,1:],end="\n\n")   # 每个维度可以使用步长跳跃切片
    print(arr[0::2,0::2])            # 多维数组取步长要用冒号


if __name__ == '__main__':
    # matrix_main()

    sile()