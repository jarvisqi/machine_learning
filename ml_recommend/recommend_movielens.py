#!/usr/bin/env python
# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from math import sqrt


def read_file():
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('data\\ml-100k\\u.data', sep='\t', names=header)
    # print(df.user_id)
    # 去重之后得到一个元祖，分别表示行与列,大小分别为943与1682
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    print('用户数量 :' + str(n_users) + ', 电影数量 :' + str(n_items))

    # 将样本分为训练集与测试集，验证集占25%
    # 根据测试样本的比例(test_size)将数据混洗并分割成两个数据集。
    train_data, test_data =train_test_split(df, test_size=0.25)

    # 创建两个矩阵  训练矩阵  验证矩阵
    train_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    # 计算user相似矩阵与item相似矩阵,大小分别为943*943,1682*1682
    # sklearn的pairwise_distances函数来计算余弦相似性
    # 相似性矩阵
    user_similar = pairwise_distances(train_data_matrix, metric="cosine")
    item_similar = pairwise_distances(train_data_matrix.T, metric="cosine")
    print(item_similar.shape[0])
    print(item_similar[1][1:11])
    print(item_similar[2][1:11])
    print(item_similar[3][1:11])
    for t in range(item_similar.shape[1]):  
        item =sorted(item_similar[t])[-10:]  
        # print(t,item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9])  

    return (train_data_matrix, test_data_matrix, user_similar, item_similar)

    # train_data_matrix, test_data_matrix, user_similar, item_similar = read_file()
    # print('user_similar.shape is :', user_similar.shape)
    # print('item_similar.shape is :', item_similar.shape)


def predict(rating, similar, type='user'):
    '''
    评分预测
    :param rating: 评分
    :param similar: 相似性
    :param type: 类型
    :return: 
    '''
    if type == 'user':
        mean_user_rating = rating.mean(axis=1)
        rating_diff = (rating - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similar.dot(
            rating_diff) / np.array([np.abs(similar).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similar) / np.array([np.abs(similar).sum(axis=1)])
    return pred


def predict_out(train_data_matrix, test_data_matrix, u_similar, i_similar):
    '''
    输入结果
    :param train_data_matrix: 训练矩阵
    :param test_data_matrix: 验证矩阵
    :param u_similar: 
    :param i_similar: 
    :return: 
    '''
    # user和item预测
    user_prediction = predict(train_data_matrix, u_similar, type='user')
    item_prediction = predict(train_data_matrix, i_similar, type='item')

    print('基于用户 CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print('基于电影 CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))


def rmse(prediction, ground_truth):
    '''
     真实值
    :param prediction: 
    :param ground_truth: 
    :return: 
    '''
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    # 均方根误差 MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度
    return sqrt(mean_squared_error(prediction, ground_truth))


def recommend_result():
    train_data_matrix, test_data_matrix, user_similar, item_similar = read_file(
    )

    print('用户相似性矩阵 :', user_similar.shape)
    print('电影相似性矩阵 :', item_similar.shape)
 
    predict_out(train_data_matrix, test_data_matrix, user_similar,
                item_similar)


if __name__ == '__main__':
    recommend_result()
