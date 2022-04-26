# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题



def check_nan():
    """
    缺失值含义分析与删除
    """

    train = pd.read_csv("./data/house/train.csv")
    na_count = train.isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(train)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'rate'])
    na_data.head(20)
    print(na_data)

    # 删除缺失值比较多的特征  删除了缺失值超过20%的特征
    train = train.drop(na_data[na_data['rate'] > 0.20].index, axis=1)
    d_count = train.isnull().sum().sort_values(ascending=False)
    print(d_count)

    return train


def load_data():
    # train = pd.read_csv("./data/house/train.csv")
    train = check_nan()
    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))
    x = train.drop(['SalePrice', 'Id'], 1)
    y = train['SalePrice']

    return x, y


def feature_selection():
    """
    特征选择
    """
    x, y = load_data()
    params = {
        # 节点的最少特征数 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
        'min_child_weight': 100,
        'eta': 0.02,        # 如同学习率 [默认0.3]
        'colsample_bytree': 0.7,   # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
        # 这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。 需要使用CV函数来进行调优。 典型值：3-10
        'max_depth': 12,
        'subsample': 0.7,   # 采样训练数据，设置为0.7
        'alpha': 1,         # L1正则化项 可以应用在很高维度的情况下，使得算法的速度更快。
        'gamma': 1,         # Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。
        'silent': 1,        # 0 打印正在运行的消息，1表示静默模式。
        'verbose_eval': True,
        'seed': 12
    }
    xgtrain = xgb.DMatrix(x, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=10)
    features = [x for x in x.columns if x not in ['SalePrice', 'id']]

    create_feature_map(features)
    # 获得每个特征的重要性
    importance = bst.get_fscore(fmap='./data/house/xgb.fmap')
    print("特征数量", len(importance))
    # 重要性排序
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=["feature", "fscore"])
    df["fscore"] = df["fscore"] / df['fscore'].sum()
    print(df)
    label = df['feature'].T.values
    xtop = df['fscore'].T.values
    idx = np.arange(len(xtop))
    fig = plt.figure(figsize=(12, 6))
    plt.barh(idx, xtop, alpha=0.8)
    plt.yticks(idx, label,)
    plt.grid(axis='x')              # 显示网格
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.title('XGBoost 特征选择图示')
    plt.show()


def create_feature_map(features):
    outfile = open('./data/house/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def feature_ref():
    x, y = load_data()

    data_col=x.columns
    # 缺失值补全
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x)
    x=imp.transform(x)
    print("2",np.isnan(x).any())
    model = SVR(kernel='linear')
    # 选取影响最大的5个特征
    rfe = RFE(model, 5)
    rfe = rfe.fit(x, y)
    for i, v in enumerate(rfe.support_):
        if v:
            print(data_col[i])

def train():

    PATH = "./data/house/"
    df_train = pd.read_csv(f'{PATH}train.csv', index_col='Id')
    df_test = pd.read_csv(f'{PATH}test.csv', index_col='Id')
    target = df_train['SalePrice']
    df_train = df_train.drop('SalePrice', axis=1)
    df_train['training_set'] = True
    df_test['training_set'] = False
    df_full = pd.concat([df_train, df_test])
    df_full = df_full.interpolate()
    df_full = pd.get_dummies(df_full)
    df_train = df_full[df_full['training_set'] == True]
    df_train = df_train.drop('training_set', axis=1)
    df_test = df_full[df_full['training_set'] == False]
    df_test = df_test.drop('training_set', axis=1)


    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rf.fit(df_train, target)
    preds = rf.predict(df_test)
    my_submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds})
    my_submission.to_csv(f'{PATH}submission.csv', index=False)

    # rmse = np.sqrt(np.mean((preds - y_test)**2))
    # print("rms:", rmse)


if __name__ == '__main__':
    
    # check_nan()

    # feature_ref()

    # feature_selection()

    train()


