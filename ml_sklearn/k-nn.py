import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import neighbors
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体  
plt.rcParams['axes.unicode_minus'] = False  #解决保存图像是负号'-'显示为方块的问题  


def main():
    """
    knn算法实现
    """
    data = []
    labels = []
    with open("data\knn.txt") as ifile:
        for line in ifile:
            tokens = line.strip().split(" ")
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    # print(data)
    x = np.array(data)
    labels = np.array(labels)
    # 构造一个0填充的矩阵
    y = np.zeros(labels.shape)
    y[labels == "fat"] = 1

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 创建网格以方便绘制 
    h = .01
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 使用knn分类器训练
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)
    # 预测
    prec = clf.predict(x)
    # 准确率 召回率
    precision, recall, thresholds = precision_recall_curve(
        y_train, clf.predict(x_train))

    # print("score", clf.score(x_train, y_train))
    # print(clf.predict_proba(x_test)[:,1])
    # answer = clf.predict_proba(x)
    y_pred = clf.predict(x_train)
    print(y_pred)
    print(y)
    # print(answer)
    # print(y_pred)
    # y，y_pred数据类型必须一直
    print(classification_report(y_train, y_pred, target_names=['thin', 'fat']))
    print("score", clf.score(x_train, y_train))
    # 将整个测试空间的分类结果用不同颜色区分开 
    answer = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    z = answer.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # 绘制训练样本
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel(u'身高')
    plt.ylabel(u'体重')
    plt.show()

    print(x_test)
    print(clf.predict(x_test))


if __name__ == '__main__':
    main()
