from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np

import scipy as sp
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    """
    数据预处理
    """

    le = LabelEncoder()
    x = [1, 2, 2, 6, 8, 12, 45, 23]
    le.fit(x)
    tsf = le.transform(x)
    print(tsf)
    #非数值型转化为数值型
    y = ["名称", "产地", "编号", "属性", "功能", "规格"]
    le.fit(y)
    str_tsf = le.transform(y)
    print(np.unique(y))
    print(str_tsf)

    # （零均值规范化）。计算方式是将特征值减去均值，除以标准差。
    print(scale(x))

    # 特征标准化
    sts = StandardScaler()
    # scaler = sts.fit(x)
    # print(scaler.transform(x))

    # 规范化将不同变化范围的值映射到相同的固定范围，常见的是[0,1]，此时也称为归一化
    x_t = [[1, -1, 2], [2, 0, 0], [0, 1, -1]]
    print(preprocessing.normalize(x_t, norm='l2'))

    
    # 文本特征抽取与向量化

    # sklearn.datasets支持从目录读取所有分类好的文本。
    # 不过目录必须按照一个文件夹一个标签名的规则放好。比如本文使用的数据集共有2个标签，一个为“net”，一个为“pos”
    movie_reviews = load_files('./data/endata')
    print(movie_reviews.data)
    print(movie_reviews.target)
    doc_train, doc_test, y_train, y_test = train_test_split(
        movie_reviews.data, movie_reviews.target, test_size=0.3)
    
    print(doc_train)
    # 词频统计
    count_vec = TfidfVectorizer(binary=False, stop_words=["english"])

    x_train= count_vec.fit_transform(doc_train)

    print(count_vec.get_feature_names())
    print(x_train.toarray())
    print(movie_reviews.target)  
    # 查看停用词
    print(count_vec.get_stop_words())




if __name__ == '__main__':
    main()