# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# plt.figure(figsize=(10, 6), dpi=90)


def sincos():
    '''
    三角函数
    '''
    
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    z = np.cos(x**2)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label="$sin(x)$", color="red", linewidth=0.2)
    ax.plot(x, z, "b--", label="cos(x^2)")
    
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("PyPlot 曲线 Demo")
    
    plt.show()


def linearplo():
    '''
    线性
    '''

    n = 256
    x = np.linspace(-np.pi, np.pi, n, endpoint=True)
    y = np.sin(2 * x)

    x1=np.linspace(-50,100,500,endpoint=True)
    y1=(x1**2)/9
    plt.plot(x1,y1)

    plt.show()

    X1=np.linspace(-4, 4, 100, endpoint = True)
    plt.plot(X1, (X1**2) / 9)


def scatterplot():
    '''
    散点图
    '''
    N=256
    x=np.random.rand(N)
    y=np.random.rand(N)

    # 普通颜色
    # plt.scatter(x,y)

    # 多种颜色
    # plt.scatter(x, y, c=['b','r','g'])

    area=np.pi * (20 * np.random.rand(N))**2  # 0 to 15 point radiuses
    # 随机颜色
    color=2 * np.pi * np.random.rand(N)
    plt.scatter(x, y, s = area, c = color, alpha = 0.5, cmap = plt.cm.hsv)
    plt.show()


def Ax3D():
    # 3D图标必须的模块，project='3d'的定义
    np.random.seed(42)

    # 采样个数500
    n_samples = 500
    dim = 3

    # 先生成一组3维正态分布数据，数据方向完全随机
    samples = np.random.multivariate_normal(
        np.zeros(dim),
        np.eye(dim),
        n_samples
    )

    # 通过把每个样本到原点距离和均匀分布吻合得到球体内均匀分布的样本
    for i in range(samples.shape[0]):
        r = np.power(np.random.random(), 1.0/3.0)
        samples[i] *= r / np.linalg.norm(samples[i])

    upper_samples = []
    lower_samples = []

    for x, y, z in samples:
        # 3x+2y-z=1作为判别平面
        if z > 3*x + 2*y - 1:
            upper_samples.append((x, y, z))
        else:
            lower_samples.append((x, y, z))

    fig = plt.figure('3D scatter plot')
    ax = fig.add_subplot(111, projection='3d')

    uppers = np.array(upper_samples)
    lowers = np.array(lower_samples)

    # 用不同颜色不同形状的图标表示平面上下的样本
    # 判别平面上半部分为红色圆点，下半部分为绿色三角
    ax.scatter(uppers[:, 0], uppers[:, 1], uppers[:, 2], c='r', marker='o')
    ax.scatter(lowers[:, 0], lowers[:, 1], lowers[:, 2], c='g', marker='^')

    plt.show()



  
def fun(x,y):  
    return np.power(x,2)+np.power(y,2)  

def Ax3DSq():
    fig1=plt.figure("3D 曲面图",figsize=(10,6))#创建一个绘图对象  
    ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)  
    X=np.arange(-2,2,0.1)  
    Y=np.arange(-2,2,0.1)#创建了从-2到2，步长为0.1的arange对象  
    #至此X,Y分别表示了取样点的横纵坐标的可能取值  
    #用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点  
    X,Y=np.meshgrid(X,Y)  
    Z=fun(X,Y)#用取样点横纵坐标去求取样点Z坐标  

    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.cm.rainbow)    #用取样点(x,y,z)去构建曲面  
    ax.set_xlabel('x label', color='r')  
    ax.set_ylabel('y label', color='g')  
    ax.set_zlabel('z label', color='b')#给三个坐标轴注明  
    plt.show()#显示模块中的所有绘图对象 


def brokenline():
    
    a = b = np.arange(0, 3, .02)
    c = np.exp(a)
    d = c[::-1]
    fig, ax = plt.subplots()
    ax.plot(a, c, 'k--', label='A',color='r')
    ax.plot(a, d, 'k:', label='B',color='b')
    ax.plot(a, c + d, 'k', label='C',color='y')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    plt.show()

if __name__ == '__main__':

    # sincos()
    # linearplo()
    # scatterplot()
    # Ax3D()
    # Ax3DSq()
    # brokenline()
