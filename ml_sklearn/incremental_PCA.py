# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

# 增量PCA

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    n_components = 2
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)
    X_ipca = ipca.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    colors = ['navy', 'turquoise', 'darkorange']

    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        plt.figure(figsize=(8, 8))
        for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
            plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                        color=color, lw=2, label=target_name)

        if "Incremental" in title:
            err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            plt.title(title + " of iris dataset\nMean absolute unsigned error "
                    "%.6f" % err)
        else:
            plt.title(title + " of iris dataset")
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.axis([-4, 4, -1.5, 1.5])

    plt.show()


if __name__ == '__main__':
    main()