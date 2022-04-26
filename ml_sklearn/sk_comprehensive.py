import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
import matplotlib.pyplot as plt


def main():
    # 创建和预测器对象：逻辑回归、主成分分析、管道
    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[("pca", pca), ("logistic", logistic)])
    # 数字数据集
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2)
    
    # 绘制并输出PCA频谱
    pca.fit(X_digits)
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([0.2, 0.2, 0.7, 0.7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis("tight")
    plt.xlabel("n_components")
    plt.ylabel("explained_variance")

    # 预测
    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)
    # 管道
    estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
    estimator.fit(X_train, y_train)

    pred = estimator.predict(X_test)
    print(pred)
    print(y_test)

    plt.axvline(estimator.best_estimator_.named_steps["pca"].n_components, linestyle=":", label="n_components chosen")
    plt.legend(prop=dict(size=12))
    plt.show()


def face_feature():
    from time import time
    import logging
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import fetch_lfw_people
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    # 在stdout中输出过程日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # 如果本地还没有Numpy数组格式的数据，则从网上下载。
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    # 图像数组的规模
    n_samples, h, w = lfw_people.images.shape

    X = lfw_people.data
    n_features = X.shape[1]

    # 人物id是预测目的标签
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    # 用分层K-Fold方法划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 在人脸数据集上计算PCA（当作无标签数据集）：无监督特征提取/维数压缩
    n_components = 150
    print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # 训练SVM分类模型
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # 在测试集上定量评估模型质量
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # 用matplotlib定量绘制预测器的评估
    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())


    # 在测试集的一部分上绘制预测结果图象
    def title(y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

    prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
    plot_gallery(X_test, prediction_titles, h, w)

    # 画出辨识度最高的特征脸
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()



if __name__ == '__main__':
    # main()

    face_feature()
