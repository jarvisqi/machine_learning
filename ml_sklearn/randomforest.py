import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt

iris = load_iris()

def rf_regressor():
    """
    随机森林回归
    """
    rf = RandomForestRegressor()
    rf.fit(iris.data[:150], iris.target[:150])

    # 挑选两个预测不相同的样本  
    instance=iris.data[[10,109]]  
    pred= rf.predict(instance)

    print("预测结果：",pred)
    print("实际结果：",iris.target[10],iris.target[109] )


def rf_classifier():
    """
    随机森林分类
    """
    rf = RandomForestClassifier(max_depth=4)
    X_test = iris.data[:120]
    y_test = iris.target[:120]
    rf.fit(X_test, y_test)

    # 用一个独立样本做预测。
    instance = iris.data[100:101]
    pred = rf.predict(instance)
    print("预测结果：", pred)
    print("实际结果：", iris.target[100:101])

    # 创建特征贡献值，用 ti.predict 可以得到预测值，偏差项和贡献值. 贡献值矩阵是一个 3D 数组
    prediction, bias, contributions = ti.predict(rf, instance)
    print("Instance", prediction)
    print("Bias (trainset mean)", bias)
    print("Feature contributions:", contributions)


if __name__ == '__main__':
    # rf_regressor()

    rf_classifier()
