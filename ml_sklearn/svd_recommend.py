#-*- coding:utf-8 -*-

import numpy as np 
import scipy as sp 


def cos_sim(inA, inB):
    """
    余弦相似度
    """
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def svd_est(dataMat, user, item):
    """
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    """
    n = dataMat.shape[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 奇异值分解
    U, Sigma, VT = np.linalg.svd(dataMat)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    sig4 = np.mat(np.eye(4) * Sigma[: 4])
    xformedItems = dataMat.T * U[:, :4] * sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = cos_sim(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        # 对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal / simTotal


def recommend(dataMat, user, n=3):

    # 对给定的用户建立一个未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'error'
    item_scores = []
    # 在未评分物品上进行循环
    for item in unratedItems:
        score = svd_est(dataMat, user, item)
        # 寻找前N个未评级物品，调用standEst()来产生该物品的预测得分，该物品的编号和估计值会放在一个元素列表itemScores中
        item_scores.append((item, score))
        # 按照估计得分，对该列表进行排序并返回。列表逆排序，第一个值就是最大值
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[: n]



if __name__ == '__main__':
    
    # 行：代表人
    # 列：代表菜肴名词
    # 值：代表人对菜肴的评分，0表示未评分
    
    data = [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
   
    d = recommend(sp.matrix(data), 2)

    print(d)
