# -*- coding: utf-8 -*-

from matplotlib import pyplot
from sklearn.datasets import load_files
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy import sparse
import numpy as np
import time

pyplot.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
pyplot.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class recommend:
    def __init__(self, path):
        '''
        读取文件 计算相似度 输出结果
        :param path: 文件路径
        '''
        start_time = time.time()

        # 数据读入
        self.data = np.loadtxt(path + 'product-ratings.txt')
        x_p = self.data[:, :2]  # 取前2列
        # y_p = self.data[:, 2]  # 取前2列
        x_p -= 1  # 0为起始索引

        #  todense 转换稠密矩阵
        #  csc_matrix 按col列压缩
        self.y = (sparse.csc_matrix((self.data[:, 2], x_p.T), dtype=np.float64))[:, :].todense()

        self.nUser, self.nItem = self.y.shape

        # self.show_plot(self.y, self.nItem, self.nUser)

        # 返回元组 train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata
        # X_train, X_test, y_train, y_test =cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)
        # train_data：所要划分的样本特征集
        # train_target：所要划分的样本结果
        # test_size：样本占比，如果是整数的话就是样本的数量 test_size=0.2 # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
        # random_state：是随机数的种子。
        # data[:, :2] ：取第3列的值， data[2:,3] ：取第3行 第4列，[:, :]：读取所有的值
        x_train, x_test, y_train, y_test = train_test_split(
            self.data[:, :2], self.data[:, 2], test_size=0.0)

        x = sparse.csc_matrix((y_train,x_train.T))

        self.item_likeness = np.zeros((self.nItem, self.nItem))

        # 训练
        for i in range(self.nItem):
            self.item_likeness[i] = self.calc_relation(x[:, i].T, x.T)
            self.item_likeness[i, i] = -1

        #  shape 读取矩阵的长度，比如shape[1]就是读取矩阵第二维度的长度。它的输入参数可以是一个整数，表示维度，也可以是一个矩阵
        for t in range(self.item_likeness.shape[1]):
            item = self.item_likeness[t].argsort()[-3:]  #取后三列

            # t = t if t > 0  else 1
            p1 = item[0]  #if item[0] > 0 else 1
            p2 = item[1]  #if item[1] > 0 else 1
            p3 = item[2]  #if item[2] > 0 else 1

            print("购买了商品 %d 的用户，推荐购买商品 %d,%d,%d " % (t, p1, p2, p3))

        print("time spent:", time.time() - start_time)

    def show_plot(self, y, xt, yt):
        '''
        可视化矩阵
        :param y:  矩阵
        :param xt: x轴数据 
        :param yt: y轴数据 
        :return: 
        '''
        pyplot.imshow(y, interpolation='nearest')
        pyplot.xlabel('商品')
        pyplot.ylabel('用户')
        pyplot.xticks(range(xt))
        pyplot.yticks(range(yt))
        pyplot.show()

    def calc_relation(self, testfor, data):
        '''
        计算向量test与data数据每一个向量的相关系数，data一行为一个向量
        :param testfor: 
        :param data: 
        :return: 
        '''
        return np.array([np.corrcoef(testfor, c)[0, 1] for c in data])

    def all_correlations(self, y, X):
        ''' 
        luispedro 提供的加速函数:
        :param y: 
        :param X: 
        :return: 
        '''
        X = np.asanyarray(X, float)
        y = np.asanyarray(y, float)
        xy = np.dot(X, y)
        y_ = y.mean()
        ys_ = y.std()
        x_ = X.mean(1)
        xs_ = X.std(1)
        n = float(len(y))
        ys_ += 1e-5  # Handle zeros in ys
        xs_ += 1e-5  # Handle zeros in x
        return (xy - x_ * y_ * n) / n / xs_ / ys_


if __name__ == '__main__':
    recommend('D:/Learning/Matlab/data/')
