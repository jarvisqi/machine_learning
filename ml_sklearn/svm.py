# -*- coding: utf-8 -*-  
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

plt.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体  
plt.rcParams['axes.unicode_minus'] = False  #解决保存图像是负号'-'显示为方块的问题  


def main():

    data = []
    labels = []
    with open("./data/tree.txt") as ifile:
        for line in ifile:
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)
    y[labels == 'fat'] = 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.0)

    h = .02
    # create a mesh to plot in  
    x_min, x_max = x_train[:, 0].min() - 0.1, x_train[:, 0].max() + 0.1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    ''''' SVM '''
    # title for the plots  
    titles = [
        'LinearSVC (linear kernel)', 'SVC with polynomial (degree 3) kernel',
        'SVC with RBF kernel', 'SVC with Sigmoid kernel'
    ]
    clf_linear = svm.SVC(kernel='linear').fit(x, y)          # 线性函数
    clf_poly = svm.SVC(kernel='poly', degree=3).fit(x, y)    # 多项式函数
    clf_rbf = svm.SVC().fit(x, y)                            # 径向基函数 
    clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x, y)        # Sigmoid函数

    for i, clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):
        answer = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        print(clf)
        print(np.mean(answer == y_train))
        print(answer)
        print(y_train)

        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Put the result into a color plot  
        z = answer.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points  
        plt.scatter(
            x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
        plt.xlabel(u'身高')
        plt.ylabel(u'体重')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

    print("\n")

    svd(x_train)

def svd(train_data_matrix):
    print(train_data_matrix.shape)
    
    u, s, vt = svds(train_data_matrix, k = 5)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
 
    print ('User-based CF MSE: ',str(rmse(X_pred, test_data_matrix)))


if __name__ == '__main__':
    main()
 