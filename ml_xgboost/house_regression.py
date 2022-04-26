# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from xgboost import XGBRegressor,plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('./data/house/train.csv')
    # 统计缺失值
    na_count = data.isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(data)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'rate'])
    # na_data.head(20)
    # print(na_data)

    # 删除缺失值比较多的特征  删除了缺失值超过20%的特征
    data = data.drop(na_data[na_data['rate'] > 0.40].index, axis=1)
    # data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    for f in data.columns:
        if data[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))

    columns = data.columns
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = imputer.fit_transform(data)
    data = imputer.transform(data)

    data = pd.DataFrame(data=data, columns=columns)
    print(data.columns)
    y = data['SalePrice']
    # select_dtypes(include=None, exclude=None)[source] include：包含某些列，exclude：排除某些列
    # X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
    X = data.drop(['SalePrice', 'Id'], 1)
    train_X, test_X, train_y, test_y = train_test_split( X.as_matrix(), y.as_matrix(), test_size=0.2)

    model = XGBRegressor(max_depth=8, n_estimators=1000, learning_rate=0.01, objective="reg:linear",
                         colsample_bylevel=.8, colsample_bytree=.8, nthread=2, seed=1024)
    model.fit(train_X, train_y, eval_set=[(test_X, test_y)], early_stopping_rounds=30, verbose=False)

    # 显示重要特征
    plot_importance(model)
    plt.show()

    predictions = model.predict(test_X)
    print("XGBoost Root Mean Squared Error : ", (np.sqrt(mean_absolute_error(predictions, test_y))))
    print("XGBoost Mean Absolute Error : ", (mean_absolute_error(predictions, test_y)))
    print("XGBoost Mean Squared Error : ", (mean_squared_error(predictions, test_y)))

    predictions = model.predict(X[1001:1005].as_matrix())
    # 实际房价预测误差几千块
    print(predictions,'\n',y[1001:1005].values)

    # rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=6)
    # rf.fit(train_X, train_y)
    # rf_predictions = rf.predict(test_X)

    # print("RandomForest Mean Absolute Error : ", str(mean_absolute_error(predictions, test_y)))
    # print("RandomForest Mean Squared Error : ", str(mean_squared_error(predictions, test_y)))


if __name__ == '__main__':
    main()
