# -*- coding:utf-8 -*-

# xgboost 多分类
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

num_round=5

def main():

    data = np.loadtxt('./data/dermatology.data', delimiter=',',
                      converters={33: lambda x: int(x == '?'), 34: lambda x: int(x) - 1})
    sz = data.shape
    train = data[:int(sz[0] * 0.7), :]
    test = data[int(sz[0] * 0.7):, :]

    train_X = train[:, :33]
    train_y = train[:, 34]

    test_X = test[:, :33]
    test_y = test[:, 34]
    print(train_X.shape, test_y.shape)

    # xg_train = xgb.DMatrix(train_X, train_y)
    # xg_test = xgb.DMatrix(test_X, test_y)
    # # 设置参数
    # params = {
    #     'objective' : 'multi:softmax',  # 用 softmax 多分类
    #     'eta' : 0.1,                    # 如同学习率 [默认0.3]
    #     'max_depth' : 6,
    #     'silent' : 1,                   # 0 打印正在运行的消息，1表示静默模式。
    #     'nthreed' : 4,
    #     'num_class' : 6
    # }
    # # 训练
    # watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    # bst = xgb.train(params, xg_train, num_boost_round=num_round, evals=watchlist)
    # # 保存
    # bst.save_model('./models/multi_class_softmax.model')

    # # 预测
    # y_pred = bst.predict(xg_test)
    # error_rate = np.sum(y_pred != test_y) / test_y.shape[0]
    # print('Test error using softmax = {}'.format(error_rate))


    # params['objective'] = 'multi:softprob'
    # bst = xgb.train(params, xg_train, num_round, watchlist)
    # bst.save_model('./models/multi_class_softprob.model')
    
    # # 预测
    # y_pred = bst.predict(xg_test)
    # pred_prob = y_pred.reshape(test_y.shape[0], 6)
    # pred_label = np.argmax(pred_prob, axis=1)
    # error_rate = np.sum(pred_label != test_y) / test_y.shape[0]
    # print('Test error using softprob = {}'.format(error_rate))

    # n_estimators 随机森林中树的数量  
    # colsample_bytree：训练每棵树时用来训练的特征的比例，
    # subsample：训练每棵树时用来训练的数据占全部的比例。用于防止 Overfitting。
    model = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, silent=True, objective="multi:softmax",
                          nthread=4, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                          reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=7, missing=None)
    eval_set = [(test_X, test_y)]
    # early_stopping_rounds：用于控制在 Out Of Sample 的验证集上连续多少个迭代的分数都没有提高后就提前终止训练。用于防止 Overfitting。
    # mlogloss :“mlogloss”: 多分类用  “merror”: Multiclass classification error rate.
    bst=model.fit(train_X,train_y,eval_set=eval_set,eval_metric=['mlogloss','merror'],early_stopping_rounds=30,verbose=True)
    y_pred = bst.predict(test_X)
    error_rate = np.sum(y_pred != test_y) / test_y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))

if __name__ == '__main__':
    main()

 
