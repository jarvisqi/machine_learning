# -*- coding:utf-8 -*-

# xgboost二分类
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

np.random.seed(19260817)
seed=7


def main():
    dataset = np.loadtxt('./data/pima-indians-diabetes.data', delimiter=',')
    # print(dataset)
    x = dataset[:, 0:8]
    y = dataset[:, 8]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    model = XGBClassifier(max_depth=10, learning_rate=0.01, n_estimators=1000,min_child_weight=6)
    model.fit(x_train, y_train)

    eval_set = [(x_test, y_test)]
    model.fit(x_train, y_train, early_stopping_rounds=30, eval_metric='logloss', eval_set=eval_set, verbose=True)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('精度：', acc)
    # plot_importance(model)
    # pyplot.show()

    # # 调参
    # # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # # param_grid = dict(learning_rate=learning_rate)
    # # skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # # 网格搜索
    # # grid_search = GridSearchCV( model, param_grid, scoring='neg_log_loss', n_jobs=1, cv=skfold)
    # # grid_result = grid_search.fit(x, y)

    # # print("best_score: %f best_params %s" % (grid_result.best_score_, grid_result.best_params_))


if __name__ == '__main__':
    main()
