# -*- coding: utf-8 -*-

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD


def main():
    # # 读入图片矩阵600*900*3
    # img_matrix = img.imread('./data/beauty.jpg')

    # plt.imshow(img_matrix)
    # plt.show()

    img_array = img.imread("./data/beauty.jpg")
    shape = img_array.shape
    # 高度、宽度、RGB通道数=3
    height, width, channels = shape[0], shape[1], shape[2]

    # 转换成numpy array
    img_matrix = np.array(img_array)

    # 存储RGB三个通道转换后的数据
    planes = []
    # RGB三个通道分别处理
    for idx in range(channels):
        # 提取通道
        plane = img_matrix[:, :, idx]
        # 转成二维矩阵
        plane = np.reshape(plane, (height, width))
        
        # 保留10个主成分
        svd = TruncatedSVD(n_components=10)
        # 拟合数据，进行矩阵分解，生成特征空间，剔去无关紧要的成分
        svd.fit(plane)
     
        # 将输入数据转换到特征空间
        new_plane = svd.transform(plane)
        # print(new_plane)
        # # 再将特征空间的数据转换会数据空间
        plane = svd.inverse_transform(new_plane)
        # 存起来
        planes.append(plane)

    # print(planes)
    # 合并三个通道平面数据
    img_matrix = np.dstack(planes)
    # 显示处理后的图像
    plt.imshow(img_matrix)

    plt.show()


if __name__ == '__main__':
    main()


