# -*- coding:utf-8 -*-
import os
import datetime
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, scale
import matplotlib.pyplot as plt

np.random.seed(19260817)
plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_time(data):
    DD = datetime.datetime.strptime(data, "%Y-%m-%d")
    day = DD.day
    month = DD.month
    return day, month


def parse_sh(data):
    if data == "0":
        return "0"
    else:
        return "1"


def parse_year(data):
    return 2015 - data


def main():
    pass


def pre_train():

    pd_train = pd.read_csv('./data/rossmann/train.csv')
    pd_test = pd.read_csv('./data/rossmann/test.csv')
    # 先统计缺失的列
    na_train = pd_train.isnull().sum().sort_values(ascending=False)
    na_test = pd_test.isnull().sum().sort_values(ascending=False)
    print(na_train)
    print(na_test)
    # Open   test 有列缺失
    pd_test['Open'].fillna(1, inplace=True)

    # 处理时间列
    pd_train['Day'], pd_train['Month'] = zip(
        *pd_train['Date'].apply(parse_time))
    pd_test['Day'], pd_test['Month'] = zip(*pd_test['Date'].apply(parse_time))

    # 处理 StateHoliday
    pd_train['SH'] = pd_train['StateHoliday'].apply(parse_sh)
    pd_test['SH'] = pd_test['StateHoliday'].apply(parse_sh)

    # 删除原来的值
    pd_train.drop(['Date', 'StateHoliday'], inplace=True, axis=1)
    pd_test.drop(['Date', 'StateHoliday'], inplace=True, axis=1)

    # 保存处理后的数据
    pd_train.to_csv("./data/rossmann/train2.csv", index=False)
    pd_test.to_csv("./data/rossmann/test2.csv", index=False)
    print("saved")

    # # 缺失值处理
    # imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    # imp.fit(train)
    # train = imp.transform(train)


def pre_store():
    pd_store = pd.read_csv('./data/rossmann/store.csv')
    # 先统计缺失的列
    na_store = pd_store.isnull().sum().sort_values(ascending=False)
    print(na_store)

    # CompetitionDistance 填充0,然后标准化
    pd_store['CompetitionDistance'].fillna(0, inplace=True)
    scale(pd_store['CompetitionDistance'], copy=False)

    # CompetitionOpenSinceYear 填充0,然后更改
    pd_store['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    pd_store['Cosyc'] = pd_store['CompetitionOpenSinceYear'].apply(parse_year)

    # StoreType/Assortment 变哑变量
    pd_store = pd.get_dummies(pd_store['StoreType'], prefix='StoreType').join(pd_store)
    pd_store = pd.get_dummies(pd_store['Assortment'], prefix='Assortment').join(pd_store)

    # 删除旧列
    pd_store.drop(['StoreType', 'Assortment', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceWeek',
                   'Promo2SinceYear', 'PromoInterval'], inplace=True, axis=1)

    # print(pd_store.head())
    pd_store.to_csv("./data/rossmann/store2.csv", index=False)


def load_data():

    pd_train = pd.read_csv('./data/rossmann/train2.csv')
    pd_test = pd.read_csv('./data/rossmann/test2.csv')
    pd_store = pd.read_csv('./data/rossmann/store2.csv')

    print(pd_train.columns)
    print(pd_test.columns)
    # # 数据合并
    if not os.path.exists("./data/rossmann/merge.csv"):
        X_train = pd.merge(pd_store, pd_train, on='Store')
        X_train.to_csv('./data/rossmann/merge.csv')

    data = pd.read_csv('./data/rossmann/merge.csv')
    x = data.drop(['Sales', 'Store'], 1)
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=7)

    return x_train, x_test, y_train, y_test


def feature_selection():
    X_train, X_test, y_train, y_test = load_data()

    print("load.......")
    print(X_train.columns)

    # booster：提升器# 有三种选择gbtree、dart、gblinear前两种为树模型，最后一个是线性模型。
    # silent：无记录# 为0代表会输出运行的信息，1代表不输出。
    # nthread：并行数目。# 提高速度的。
    # num_pbuffer：程序自动设置，缓冲的大小。记录上一轮预测的结果。# 这是由程序自动设置的，用户无须管。
    # num_feature：程序自动设置，特征维度的大小。# 这是由程序自动设置的，用户无须管。

    # eta：学习率。
    # gamma：最小损失分裂。 #如果比较小，那么叶子节点就会不断分割，越具体细致。如果比较大，那么算法就越保守。
    # max_depth：最大树的深度。# 数值越大，模型越复杂、越具体，越有可能过拟合过。
    # min_child_weight：子节点最小的权重。# 重要参数。如果叶子节点切分低于这个阈值，就会停止往下切分的了。该参数数值越大，就越保守，越不会过拟合。与gamma有相关性，但是gamma关注损失，而这关心自身权重。
    # max_delta_step：最大delta的步数# 数值越大，越保守。一般这个参数是不需要的，但它能帮助语料极度不平衡的逻辑回归模型。
    # subsample：子样本数目 # 是否只使用部分的样本进行训练，这可以避免过拟合化。默认为1，即全部用作训练。
    # colsample_bytree：每棵树的列数（特征数）。# 默认为1
    # colsample_bylevel：每一层的列数（特征数）。#默认为1
    # lambda：L2正则化的权重。# 增加该数值，模型更加保守。
    # alpha ：L1正则化的权重。# 增加该数值，模型更加保守。
    # tree_method：树构造的算法。# 默认auto，使用启发式选择最快的一种。如果是中小型数据，就用exact 准确算法，大型数据就选择approx 近似算法。
    # sketch_eps：# 只用在approx 算法的，用户一般不用调。调小可以获得更准确的序列。
    # scale_pos_weight：用在不均衡的分类。# 默认为1，还有一种经典的取值是： sum(negative cases) / sum(positive cases)
    # updater：更新器。# 如果更新树，是一个高级参数。程序会自动选择。当然用户也能自己选择。只是有很多种选择，我看不懂具体的。
    # refresh_leaf：# 当updater= refresh才有用。设置为False时，叶子节点不更新，只更新中间节点。
    # process_type：程序运行方式。# 两种选择，默认是default，新建一些树。而选择update，代表基于已有的树，并对已有的树进行更新。

    # 飞镖提升器的参数 Dart Booster
    #     sample_type：选样方式。# 两种选择。uniform 代表一样的树进行舍弃，weighted 代表选择权重比例的树进行舍弃。
    #     normalize_type：正则化方式。# 两种选择。tree、forest，区别在于计算新树的权重、舍弃树的计算，两种公式不同。
    #     rate_drop：舍弃的比例。# 舍弃上一轮树的比例。
    #     one_drop：# 当值不为0的时候，至少有一棵树被舍弃。
    #     skip_drop：有多少的概率跳过舍弃树的程序。

    # 线性提升器的参数 Linear Booster（只有3个）
    #     lambda：L2正则化的权重。# 增加该数值，模型更加保守。
    #     alpha ：L1正则化的权重。# 增加该数值，模型更加保守。
    #     lambda_bias：L2正则化的偏爱。# 不知道数值大小的影响作用。

    # 任务参数

    #     objective：reg:linear线性回归、reg:logistic逻辑回归、binary:logistic逻辑回归处理二分类（输出概率）、
    #     binary:logitraw逻辑回归处理二分类（输出分数）、count:poisson泊松回归处理计算数据（输出均值、max_delta_step参数默认为0.7）
    #     multi:softmax多分类（需要设定类别的个数num_class）、multi:softprob多分类（与左侧的一样，只是它的输出是ndata*nclass，也就是输出输入各类的概率）
    #     rank:pairwise处理排位问题、reg:gamma用γ回归（返回均值）、reg:tweedie用特威迪回归。
    #     base_score：初始化时，会预测各种类别的分数。# 当迭代足够多的次数，改变这个值是没有什么用的。
    #     eval_metric：评估分数的机制。# 用户可以加新评估机制进来的。主要的：rmse针对回归问题、error针对分类问题、map（Mean average precision）针对排位问题。
    #     seed：种子。# 默认为0。随机数的意思。

    # use_buffer：数据加载的缓冲大小。
    # num_round：运行的次数。

    params = {
        # 节点的最少特征数 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
        'min_child_weight': 60,
        'eta': 0.02,        # 如同学习率 [默认0.3]
        'colsample_bytree': 0.7,   # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
        # 这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。 需要使用CV函数来进行调优。 典型值：3-10
        'max_depth': 7,
        'subsample': 0.7,   # 采样训练数据，设置为0.7
        'alpha': 1,         # L1正则化项 可以应用在很高维度的情况下，使得算法的速度更快。
        'gamma': 1,         # Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。
        'silent': 1,        # 0 打印正在运行的消息，1表示静默模式。
        'verbose_eval': True,
        'seed': 12
    }

    xgtrain = xgb.DMatrix(X_train, label=y_train)

    # params (dict)：参数。
    # dtrain (DMatrix)：给XGB训练用的数据。
    # num_boost_round (int)：运行多少次迭代。
    # evals (list of pairs (DMatrix, string))：这是由pari元素组成的list，可显示性能出来的验证集。
    # obj (function)：自定制的目标函数。
    # feval (function)：自定制的评估函数。
    # maximize (bool)：是否要验证集得分最大。
    # early_stopping_rounds (int)：可在一定的迭代次数内准确率没有提升（evals列表的全部验证集都没有提升）就停止训练。# 使用 bst.best_ntree_limit 可以得到真实的分数。
    # evals_result (dict)：通过字典找特定验证集evals 分数。# 例如验证集evals = [(dtest,’eval’), (dtrain,’train’)]，并且在params中设定，{‘eval_metric’: ‘logloss’}。就可根据str找分数 {‘train’: {‘logloss’: [‘0.48253’, ‘0.35953’]}, ‘eval’: {‘logloss’: [‘0.480385’, ‘0.357756’]}}
    # verbose_eval (bool or int)：是否显示每次迭代的情况。# 如果设置为True就代表每次都显示，False都不显示，一个正整数（如4）就代表每4轮输出一次。
    # learning_rates (list or function (弃用 - use callback API instead))# 如果是一个list，就代表每一轮的学习率是多少。此外params中也有一个学习率eta，但是eta只能是一个浮点数，不够这个具体。
    # xgb_model (file name of stored xgb model or ‘Booster’ instance)：通过名字加载已有的XGB模型。
    # callbacks (list of callback functions)：在每次迭代后的处理。是一个list类型。# 可以预设定参数（暂时不会用） [xgb.callback.reset_learning_rate(custom_rates)]

    bst = xgb.train(params, xgtrain, num_boost_round=10)
    # 特征
    features = [x for x in X_train.columns if x not in ['Sales', 'Store']]
    create_feature_map(features)
    # 获得每个特征的重要性
    importance = bst.get_fscore(fmap='./data/rossmann/xgb.fmap')
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



def transform_data(train):
    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

    return train


def create_feature_map(features):
    outfile = open('./data/rossmann/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def train():

    X_train, X_test, y_train, y_test = load_data()
    eval_set = [(X_test, y_test)]
    print(X_train.shape)
    print(y_train.shape)

    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=500, objective="reg:linear",
                             nthread=4, silent=True, subsample=0.8, colsample_bytree=0.8)
    bst = model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=30, verbose=True)

    # 显示重要特征
    xgb.plot_importance(model)
    plt.show()

    # preds = bst.predict(X_test)
    # rmse = np.sqrt(np.mean((preds - y_test)**2))
    # print("rms:", rmse)


if __name__ == '__main__':
    # main()
    # pre_train()
    # pre_store()
    feature_selection()
    # load_data()

    # train()
