# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
np.random.seed(19260817)


def d2():
    # x轴采样点
    x = np.linspace(0, 5, 100)
    # 通过下面曲线加上噪声生成数据，所以拟合模型用y
    y = 2 * np.sin(x) + 0.3 * x**2
    y_data = y + np.random.normal(scale=0.3, size=100)

    # 指定 figure 图表名称
    plt.figure('data')
    #  '.' 标明画散点图 每个散点的形状是园
    plt.plot(x, y_data, '.')

    # 画模型的图，plot函数默认画连线图
    plt.figure('model')
    plt.plot(x, y)
    # 两个图画一起
    plt.figure('data & model')
    # 通过'k'指定线的颜色，lw指定线的宽度
    # 第三个参数除了颜色也可以指定线形，比如'r--'表示红色虚线
    # 更多属性可以参考官网：http://matplotlib.org/api/pyplot_api.html
    plt.plot(x, y, 'k', lw=2)
    # scatter可以更容易地生成散点图
    plt.scatter(x, y_data)
    # 保存当前图片
    plt.savefig('./data/result.png')
    # 显示图像
    plt.show()


def histogram():
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.major.size'] = 0
    mpl.rcParams['ytick.major.size'] = 0

    # 包含了狗，猫和猎豹的最高奔跑速度，还有对应的可视化颜色
    speed_map = {
        'dog': (48, '#7199cf'),
        'cat': (45, '#4fc4aa'),
        'cheetah': (120, '#e1a7a2')
    }

    fig = plt.figure("Bar chart & Pie chart")
    # 在整张图上加入一个子图，121的意思是在一个1行2列的子图中的第一张
    ax = fig.add_subplot(121)
    ax.set_title('Running speed - bar chart')
    # 生成x轴每个元素的位置
    xticks = np.arange(3)
    # 定义柱状图的宽度
    bar_width = 0.5
    # 动物名字
    animals = speed_map.keys()
    # 速度
    speeds = [x[0] for x in speed_map.values()]
    # 颜色
    colors = [x[1] for x in speed_map.values()]
    # 画柱状图，横轴是动物标签的位置，纵轴是速度，定义柱的宽度，同时设置柱的边缘为透明
    # xticks + bar_width / 2 柱位置在刻度中央
    bars = ax.bar(xticks + bar_width / 2, speeds, width=bar_width, edgecolor='none')
    # 设置y轴的标题
    ax.set_ylabel('Speed(km/h)')
    # x轴每个标签的具体位置，设置为每个柱的中央
    ax.set_xticks(xticks + bar_width / 2)
    # 设置每个标签的名字
    ax.set_xticklabels(animals)
    # 设置x轴的范围
    ax.set_xlim([bar_width / 2 - 0.5, 3 - bar_width / 2])
    # 设置y轴的范围
    ax.set_ylim([0, 125])

    # 给每个bar分配指定的颜色
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 在122位置加入新的图
    ax = fig.add_subplot(122)
    ax.set_title('Running speed - pie chart')

    # 生成同时包含名称和速度的标签
    labels = ['{}\n{} km/h'.format(animal, speed) for animal, speed in zip(animals, speeds)]

    # 画饼状图，并指定标签和对应颜色
    ax.pie(speeds, labels=labels, colors=colors)

    plt.show()


def d3():
    n_grids = 51        	# x-y平面的格点数 
    c =int(n_grids / 2)      	# 中心位置
    nf = 2              	# 低频成分的个数
    # 生成格点
    x = np.linspace(0, 1, n_grids)
    y = np.linspace(0, 1, n_grids)
    # x和y是长度为n_grids的array
    # meshgrid会把x和y组合成n_grids*n_grids的array，X和Y对应位置就是所有格点的坐标
    X, Y = np.meshgrid(x, y)
    # 生成一个0值的傅里叶谱
    spectrum = np.zeros((n_grids, n_grids), dtype=np.complex)    
    # 生成一段噪音，长度是(2*nf+1)**2/2
    t=int((2*nf+1)**2/2)
    noise = [np.complex(x, y) for x, y in np.random.uniform(-1.0,1.0,(t, 2))]
    # 傅里叶频谱的每一项和其共轭关于中心对称
    noisy_block = np.concatenate((noise, [0j], np.conjugate(noise[::-1])))
    # 将生成的频谱作为低频成分
    
    spectrum[c-nf:c+nf+1, c-nf:c+nf+1] = noisy_block.reshape((2*nf+1, 2*nf+1))
    # 进行反傅里叶变换
    Z = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    # 创建图表
    fig = plt.figure('3D surface & wire')
    # 第一个子图，surface图
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # alpha定义透明度，cmap是color map
    # rstride和cstride是两个方向上的采样，越小越精细，lw是线宽
    ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)
    # 第二个子图，网线图
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, lw=0.5)

    plt.show()


def dynamic():
    """
    动态图
    """
    fig,ax=plt.subplots()
    y1=[]
    for i in range(50):
        y1.append(i)
        ax.cla()
        ax.bar(y1,label='test',height=y1,width=0.3)
        ax.legend()
        plt.pause(0.2)


def draw_normal():
    # 绘制普通图像
    x = np.linspace(-1, 1, 50)
    y1 = 2 * x + 1
    y2 = x**2

    plt.figure()
    # 在绘制时设置lable, 逗号是必须的
    l1, = plt.plot(x, y1)
    l2, = plt.plot(x, y2, color='red',linewidth=1.0, linestyle='--')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    # 设置legend
    plt.legend(handles=[l1, l2, ], labels=['a', 'b'], loc='upper left')

    plt.show()


def linaer():
    # data = np.arange(100, 201)
    # plt.plot(data)
    # plt.show()

    # data = np.arange(100, 201)
    # plt.plot(data)

    # data2 = np.arange(200, 301)
    # plt.figure()
    # plt.plot(data2)
    # plt.show()

    # plot函数的第一个数组是横轴的值，第二个数组是纵轴的值，所以它们一个是直线，一个是折线；
    # data = np.arange(100, 201)
    # plt.subplot(2, 1, 1)
    # plt.plot(data)

    # data2 = np.arange(200, 301)
    # plt.subplot(2, 1, 2)
    # plt.plot(data2)

    # plt.show()

    # 参数c表示点的颜色，s是点的大小，alpha是透明度
    N = 20
    plt.scatter(np.random.rand(N) * 100, np.random.rand(N)* 100, c='r', s=100, alpha=0.5)
    plt.scatter(np.random.rand(N) * 100, np.random.rand(N)* 100, c='g', s=200, alpha=0.5)
    plt.scatter(np.random.rand(N) * 100, np.random.rand(N)* 100, c='b', s=300, alpha=0.5)
    plt.show()


def pie():
    """
    饼图  显示两个图
    """
    labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    data = np.random.rand(7)*100

    # plt.subplot(2, 1, 1)  # 2行1列subplot中的第1个subplot。

    plt.subplot(1, 2, 1)
    plt.pie(data, labels=labels, autopct='%1.1f%%')
    plt.axis("equal")
    plt.legend()

    data1 = np.random.rand(7)*100
    plt.subplot(1, 2, 2)
    # autopct指定了数值的精度格式
    plt.pie(data1, labels=labels, autopct='%1.1f%%')   
    plt.axis("equal")
    plt.legend()

    plt.show()


def bar():
    """
    条形图
    """
    # np.random.rand(N * 3).reshape(N, -1)表示先生成21（N x 3）个随机数，然后将它们组装成7行，那么每行就是三个数，这对应了颜色的三个组成部分。
    N = 7
    x = np.arange(N)
    data = np.random.randint(low=0, high=100, size=N)
    colors = np.random.rand(N * 3).reshape(N, -1)
    labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    plt.title("Weekday Data")
    plt.bar(x, data, alpha=0.8, color=colors, tick_label=labels)
    plt.show()


def hist():
    """
    直方图
    """
    # 生成了包含了三个数组的数组
    data = [np.random.randint(0, n, n) for n in [3000, 4000, 5000]]
    labels = ['3K', '4K', '5K']
    # bins数组用来指定我们显示的直方图的边界，即：[0, 100) 会有一个数据点，[100, 500)会有一个数据点
    bins = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]
    plt.hist(data, bins=bins, label=labels)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # linaer()
    # d2()
    # histogram()
    # d3()
    # draw_normal()
    # dynamic()
    # pie()
    # bar()
    hist()
