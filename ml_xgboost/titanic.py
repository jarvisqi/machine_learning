# -*- coding:utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import Imputer, LabelEncoder


def data_preprocessing():
    """
    数据预处理
    """

    pd_train = pd.read_csv('./data/titanic/train.csv')
    pd_test = pd.read_csv('./data/titanic/test.csv')
    pd_gender = pd.read_csv('./data/titanic/gender_submission.csv')
    print(pd_train.shape, pd_test.shape)

    sex_count = pd_train.groupby(['Sex', 'Survived'])['Survived'].count()
    print(sex_count)
    
    # 性别 将性别字段Sex中的值 female用0，male用1代替,类型 int
    pd_train['Sex'] = pd_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # print(pd_test.columns)
    embark_dummies  = pd.get_dummies(pd_train['Embarked'])
    pd_train = pd_train.join(embark_dummies)
    pd_train.drop(['Embarked','PassengerId'], axis=1,inplace=True)

    pd_train['Fare_Category'] = pd_train['Fare'].map(fare_category)

    columns=pd_train.columns

    # 将类型变量转换位连续变量
    for f in pd_train.columns:
        if pd_train[f].dtype == 'object':
            label = LabelEncoder()
            label.fit(list(pd_train[f].values))
            pd_train[f] = label.transform(list(pd_train[f].values))

    # 统计缺失的列
    print("统计缺失的列")
    na_train = pd_train.isnull().sum().sort_values(ascending=False)
    print(na_train)

    # 使用均值填充缺失值
    train_data= pd_train.values
    imput = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imput = imput.fit(train_data)
    train_data = imput.fit_transform(train_data)

    pd_train = pd.DataFrame(train_data, index=None, columns=columns)
    na_train = pd_train.isnull().sum().sort_values(ascending=False)
    # print("缺失值处理后：")
    # print(na_train)
    # print(pd_train.head())

    # 保存新数据
    pd_train.to_csv('./data/titanic/new_train.csv')
    pd_train.to_csv('./data/titanic/new_test.csv')


def fare_category(fare):
        if fare <= 4:
            return 0
        elif fare <= 10:
            return 1
        elif fare <= 30:
            return 2
        elif fare <= 45:
            return 3
        else:
            return 4

def load_data():
    train_data = pd.read_csv('./data/titanic/new_train.csv')
    test_data = pd.read_csv('./data/titanic/new_test.csv')

    X = train_data.drop(['Survived'], 1)
    y = train_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    return X_train, X_test, y_train, y_test


def train_logreistic():
    """
    逻辑回归
    """
    X_train, X_test, y_train, y_test = load_data()

    model = LogisticRegression(penalty='l2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rfc_rate, rmse = calc_accuracy(y_pred, y_test)
    total = total_survival(y_pred)

    return rfc_rate, rmse, total


def train_randomForster():

    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(n_estimators=500,max_depth=6,random_state=7)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    rfc_rate, rmse = calc_accuracy(y_pred, y_test)
    total = total_survival(y_pred)
    # RandomForestClassifier acc_rate：82.6816,RMS:0.4162,存活：54
    return rfc_rate, rmse, total


def train_XGBoost():
    X_train, X_test, y_train, y_test = load_data()

    model = xgb.XGBClassifier(max_depth=8, learning_rate=0.06, n_estimators=100, objective="binary:logistic",
                              silent=True,min_child_weight=6)

    eval_data = [(X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_data, early_stopping_rounds=30)

    y_pred = model.predict(X_test)
    rfc_rate, rmse = calc_accuracy(y_pred, y_test)
    total = total_survival(y_pred)

    # XGBClassifier acc_rate：80.4469,RMS:0.4422,存活：56
    return rfc_rate, rmse, total


    
def calc_accuracy(y_pred, y_true):
    """
    计算精度
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # rmse=np.sqrt(metrics.mean_squared_error(y_pred,y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    return accuracy, rmse


def total_survival(y_pred):
    """
    存活人数
    """
    total = 0
    for value in y_pred:
        if value == 1:
            total += 1
    
    return total


def train():
    
    lg_rate, lg_rmse, lg_total = train_logreistic()
    rf_rate, rf_rmse, rf_total = train_randomForster()
    xg_rate, xg_rmse, xg_total = train_XGBoost()

    print("LogisticRegression acc_rate：{0:.4f},RMS:{1:.4f},存活：{2}".format( lg_rate, lg_rmse, lg_total))
    print("RandomForestClassifier acc_rate：{0:.4f},RMS:{1:.4f},存活：{2}".format(rf_rate, rf_rmse, rf_total))
    print("XGBClassifier acc_rate：{0:.4f},RMS:{1:.4f},存活：{2}".format(xg_rate, xg_rmse, xg_total))

    # size = 3
    # total_width, n = 0.8, 3
    # width = total_width / n
    # x = np.arange(size)
    # x = x - (total_width - width) / 2
    # a = [lg_rate, rf_rate, xg_rate]
    # b = [lg_rmse, rf_rmse, xg_rmse]
    # c = [lg_total, rf_total, xg_total]
    # plt.bar(x, a,  width=width, label='a')
    # plt.bar(x + width, b, width=width, label='b')
    # plt.bar(x + 2 * width, c, width=width, label='c')
    # plt.legend()
    # plt.show()


def optimized_XGBoost():
    X_train, X_test, y_train, y_test = load_data()
    model = xgb.XGBClassifier(max_depth=7, n_estimators=300, min_child_weight=5,
                              colsample_bytree=0.6, subsample=0.9,reg_alpha=0.005)
    eval_data = [(X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_data, early_stopping_rounds=30)
    y_pred = model.predict(X_test)
    acc, rmse = calc_accuracy(y_pred, y_test)

    print("accuracy：{0:.2f}%".format(100*acc))


def cv_XGBoost():
    X_train, X_test, y_train, y_test = load_data()
    model = xgb.XGBClassifier(max_depth=7, n_estimators=300, min_child_weight=5,
                              colsample_bytree=0.6, subsample=0.9,reg_alpha=0.005)
    grid_params = {
        # 'learning_rate': np.linspace(0.01, 1, 100),
        # 'n_estimators': list(range(100, 1001, 100)),
        # 'max_depth': list(range(3, 15, 1)),
        # 'min_child_weight': list(range(1, 6, 2)),
        # 'subsample':[i/10.0 for i in range(1,10)],
        # 'colsample_bytree':[i/10.0 for i in range(1,10)],
        # 'gamma': [i/10.0 for i in range(0, 5)]
        'reg_alpha':[0.001, 0.005, 0.01, 0.05,0.1,0.2,0.3,0.4,0.5,0.6]
    }
    gcv = GridSearchCV(estimator=model, param_grid=grid_params)
    gcv.fit(X_train, y_train)

    print(gcv.best_params_)
    print("Accuracy:{0:.4f}%".format(100*gcv.best_score_))




if __name__ == '__main__':
    
    # data_preprocessing()
    # load_data()

    # train()

    # train_XGBoostCV()

    # cv_XGBoost()
    
    optimized_XGBoost()
