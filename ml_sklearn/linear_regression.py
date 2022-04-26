import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import sparse

# 机器学习中的预测问题通常分为2类：回归与分类。
# 简单的说回归就是预测数值，而分类是给数据打上标签归类。

def showplt():
    # arange 根据start与stop指定的范围以及step设定的步长，生成一个 ndarray
    x = np.arange(0, 1, 0.002)

    # scipy.stats.norm.rvs正态分布
    # （1）作用：构造正态分布的数据
    # （2）函数：scipy.stats.norm.rvs(size=100,loc=0,scal=1)
    # scipy.norm正态分布模块的随机变量函数rvs，size=100是样本大小，loc＝0是均值，scal=1是标准差
    y = norm.rvs(loc=0, size=500, scale=0.1)
    y = y + x**2
    # scatter 散点图
    plt.scatter(x, y)
    plt.show()


# 均方误差根
def rmse(y_test, y):
    return sp.sqrt(np.mean((y_test - y)**2))


# R-平方  与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档
def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / (
        (y_true - y_true.mean())**2).sum()


# 这是Conway&White《机器学习使用案例解析》里的版本
def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)


def main():
    """
    线性回归以及数据过拟合
    """
    plt.scatter(x, y, s=5)
    degree = [1, 2, 100]
    y_test = []
    y_test = np.array(y_test)

    for d in degree:
        # PolynomialFeatures 	多项式数据转换
        clf = Pipeline([('poly', PolynomialFeatures(degree=d)),('linear', linear_model.LinearRegression(fit_intercept=False))])
        # 训练
        clf.fit(x[:, np.newaxis], y)
        # 预测 np.newaxis 为 numpy.ndarray（多维数组）增加一个轴  np.newaxis 在使用和功能上等价于 None，其实就是 None 的一个别名。
        y_test = clf.predict(x[:, np.newaxis])
        print(clf.named_steps['linear'].coef_)
        print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f' %
              (rmse(y_test, y), R2(y_test, y), R22(y_test, y),
               clf.score(x[:, np.newaxis], y)))
        plt.plot(x, y_test, linewidth=2)

    plt.grid()
    plt.legend(['1', '2', '100'], loc='upper left')
    plt.show()


# 线性回归

def liner_simple():
    print("线性回归")
    x = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    reg = linear_model.LinearRegression(
        copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    reg.fit(x, y)
    print(reg.coef_)


def example():

    diabetes = datasets.load_diabetes()
    data = diabetes.data[:, np.newaxis, 2]
    print(data)
    X_train = data[:-20]
    X_test = data[-20:]
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    print('Coefficients: ', regr.coef_)
    print("mse: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('score: %.2f' % r2_score(y_test, y_pred))

    # Plot outputs
    show_scatter(X_test,y_test,y_pred)


def ridge_reg():
    """
    岭回归
    """
    diabetes = datasets.load_diabetes()
    data = diabetes.data[:, np.newaxis, 2]
    X_train = data[:-20]
    X_test = data[-20:]
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]
    reg = linear_model.Ridge(alpha=0.5, copy_X=True, fit_intercept=True,
                             max_iter=None, normalize=False, random_state=None, solver="auto", tol=1e-3)
    reg.fit(X_train, y_train)
    print("系数：", reg.coef_)
    print("系数：", reg.intercept_)
    y_pred = reg.predict(X_test)

    show_scatter(X_test,y_test,y_pred)


def lasso_reg():
    """
    锁套回归  
    是估计稀疏系数的线性模型。 它在一些情况下是有用的，因为它倾向于使用具有较少参数值的情况，
    有效地减少给定解决方案所依赖变量的数量。 因此，Lasso及其变体是压缩感知领域的基础。
    """
    diabetes = datasets.load_diabetes()
    data = diabetes.data[:, np.newaxis, 2]
    X_train = data[:-20]
    X_test = data[-20:]
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]

    reg = linear_model.Lasso(alpha=0.5,fit_intercept=True,normalize=True,precompute=False,copy_X=True,max_iter=1000,tol=1e-4)
    reg.fit(X_train, y_train)
    print("系数：", reg.coef_)
    print("系数：", reg.intercept_)
    y_pred = reg.predict(X_test)

    show_scatter(X_test,y_test,y_pred)


def loss_example():
        
    def _weights(x, dx=1, orig=0):
        x = np.ravel(x)
        floor_x = np.floor((x - orig) / dx)
        alpha = (x - orig - floor_x * dx) / dx
        return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

    def _generate_center_coordinates(l_x):
        X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
        center = l_x / 2.
        X += 0.5 - center
        Y += 0.5 - center
        return X, Y


    def build_projection_operator(l_x, n_dir):

        X, Y = _generate_center_coordinates(l_x)
        angles = np.linspace(0, np.pi, n_dir, endpoint=False)
        data_inds, weights, camera_inds = [], [], []
        data_unravel_indices = np.arange(l_x ** 2)
        data_unravel_indices = np.hstack((data_unravel_indices,
                                        data_unravel_indices))
        for i, angle in enumerate(angles):
            Xrot = np.cos(angle) * X - np.sin(angle) * Y
            inds, w = _weights(Xrot, dx=1, orig=X.min())
            mask = np.logical_and(inds >= 0, inds < l_x)
            weights += list(w[mask])
            camera_inds += list(inds[mask] + i * l_x)
            data_inds += list(data_unravel_indices[mask])
        proj_operator =sp.sparse.coo_matrix((weights, (camera_inds, data_inds)))
        return proj_operator


    def generate_synthetic_data():
        """ Synthetic binary data """
        rs = np.random.RandomState(0)
        n_pts = 36
        x, y = np.ogrid[0:l, 0:l]
        mask_outer = (x - l / 2.) ** 2 + (y - l / 2.) ** 2 < (l / 2.) ** 2
        mask = np.zeros((l, l))
        points = l * rs.rand(2, n_pts)
        mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
        res = np.logical_and(mask > mask.mean(), mask_outer)
        return np.logical_xor(res, ndimage.binary_erosion(res))

    l = 128
    proj_operator = build_projection_operator(l, l / 7.)
    data = generate_synthetic_data()
    proj = proj_operator * data.ravel()[:, np.newaxis]
    proj += 0.15 * np.random.randn(*proj.shape)


    rgr_ridge =linear_model.Ridge(alpha=0.2)
    rgr_ridge.fit(proj_operator, proj.ravel())
    rec_l2 = rgr_ridge.coef_.reshape(l, l)

    rgr_lasso =linear_model.Lasso(alpha=0.001)
    rgr_lasso.fit(proj_operator, proj.ravel())
    rec_l1 = rgr_lasso.coef_.reshape(l, l)

    plt.figure(figsize=(8, 3.3))
    plt.subplot(131)
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.title('original image')
    plt.subplot(132)
    plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('L2 penalization')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('L1 penalization')
    plt.axis('off')
    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,right=1)
    plt.show()


def multiTaskLasso_reg():
    """
    多任务 lasso  多元回归稀疏系数的线性模型
    y 是一个 (n_samples, n_tasks) 的二维数组，其约束条件和其他回归问题（也称为任务）是一样的，都是所选的特征值。
    """
    
    rng = np.random.RandomState(42)
    n_samples, n_features, n_tasks = 100, 30, 40
    n_relevant_features = 5
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    for k in range(n_relevant_features):
        coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))

    X = rng.randn(n_samples, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)

    coef_lasso_ = np.array([linear_model.Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
    coef_multi_task_lasso_ =linear_model.MultiTaskLasso(alpha=1.).fit(X, Y).coef_

    fig = plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.spy(coef_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'Lasso')
    plt.subplot(1, 2, 2)
    plt.spy(coef_multi_task_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'MultiTaskLasso')
    fig.suptitle('Coefficient non-zero location')

    feature_to_plot = 0
    plt.figure()
    lw = 2
    plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,label='Truth Value')
    plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue', linewidth=lw,label='Lasso')
    plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold', linewidth=lw,label='MultiTaskLasso')
    plt.legend(loc='upper center')
    plt.axis('tight')
    plt.ylim([-1.1, 1.1])
    plt.show()


def show_scatter(X_test,y_test,y_pred):
    # Plot outputs
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()



def diabetes_reg():
    
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    print('Input Values')
    print(diabetes_X_test)

    diabetes_y_pred = regr.predict(diabetes_X_test)
    print("Predicted Output Values")
    print(diabetes_y_pred)

    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=1)

    plt.show()


if __name__ == '__main__':
    # main()
    # run()

    # liner_simple()
    # example()
    # ridge_reg()
    # lasso_reg()
    # loss_example()
    # multiTaskLasso_reg()
    
    diabetes_reg()
