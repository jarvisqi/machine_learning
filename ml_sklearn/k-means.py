from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(19260817)
digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
sample_size = 300

print("n_digits:{},\t n_samples:{},\t n_features:{}".format(n_digits,
                                                            n_samples,
                                                            n_features))


def main():

    bench_k_means(KMeans(init="k-means++", n_clusters=n_digits, n_init=10),
                  name="k-means++", data=data)

    bench_k_means(KMeans(init="random", n_clusters=n_digits, n_init=10),
                  name="random", data=data)

    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits,
                         n_init=1), name="PCA-based", data=data)

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    h = .02
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
                s=169, linewidths=3, color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (name, (time() - t0), estimator.inertia_,
                                                                   metrics.homogeneity_score(
                                                                       labels, estimator.labels_),
                                                                   metrics.completeness_score(
                                                                       labels, estimator.labels_),
                                                                   metrics.v_measure_score(
                                                                       labels, estimator.labels_),
                                                                   metrics.adjusted_rand_score(
                                                                       labels, estimator.labels_),
                                                                   metrics.adjusted_mutual_info_score(
                                                                       labels,  estimator.labels_),
                                                                   metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size)))


def simple_kmeans():
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    # 创建聚类器对象，指定分组数量为3，并将结果每隔10个数据赋值予变量y_iris_k_means
    k_means = KMeans(n_clusters=3)
    k_means.fit(X_iris)
    y_iris_k_means = k_means.labels_[::10]

    # 分别输出聚类算法结果，与标签进行对比
    print("K-means聚类结果：")
    print(y_iris_k_means)
    print("数据集原始标签：")
    print(y_iris[::10])


def blobs():
    from sklearn.datasets.samples_generator import make_blobs

    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
    X, y = make_blobs(n_samples=1000, n_features=2,
                      centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)
    plt.scatter(X[:, 0], X[:, 1], marker="o")
    plt.show()

    # 聚类
    k1 = KMeans(n_clusters=2, random_state=9)
    y_1 = k1.fit_predict(X)
    m1 = metrics.calinski_harabaz_score(X, y_1)

    k2 = KMeans(n_clusters=3, random_state=9)
    y_2 = k2.fit_predict(X)
    m2 = metrics.calinski_harabaz_score(X, y_2)

    k3 = KMeans(n_clusters=4, random_state=9)
    y_3 = k3.fit_predict(X)
    m3 = metrics.calinski_harabaz_score(X, y_3)

    k4 = KMeans(n_clusters=5, random_state=9)
    y_4 = k4.fit_predict(X)
    m4 = metrics.calinski_harabaz_score(X, y_4)
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.show()

    plt.subplot(2, 2, 1)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    plt.scatter(X[:, 0], X[:, 1], c=y_1)
    plt.title("n_clusters :2 / h_score :{:.4f}".format(m1))

    plt.subplot(2, 2, 2)  # 两行两列,这是第二个图
    plt.scatter(X[:, 0], X[:, 1], c=y_2)
    plt.title("n_clusters :3 / h_score :{:.4f}".format(m2))

    plt.subplot(2, 2, 3)  # 两行两列,这是第三个图
    plt.scatter(X[:, 0], X[:, 1], c=y_3)
    plt.title("n_clusters :4 / h_score :{:.4f}".format(m3))

    plt.subplot(2, 2, 4)  # 两行两列,这是第四个图
    plt.scatter(X[:, 0], X[:, 1], c=y_4)
    plt.title("n_clusters :5 / h_score :{:.4f}".format(m4))

    plt.show()


if __name__ == '__main__':
    # main()

    blobs()

    # simple_kmeans()
