import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import neighbors
from sklearn.externals import joblib
import os


def main():
    """
    朴素贝叶斯实现
    """
    # 加载数据
    movies_reviews = load_files("./data/tokens")
    sp.save('./data/movie_data.npy', movies_reviews.data)
    sp.save('./data/movie_target.npy', movies_reviews.target)

    movie_data = sp.load('./data/movie_data.npy')
    movie_target = sp.load('./data/movie_target.npy')
    x = movie_data
    y = movie_target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words="english")
    # 训练数据
    x_train = count_vec.fit_transform(x_train)
    x_test = count_vec.transform(x_test)

    # 分类器   
    clf = MultinomialNB().fit(x_train, y_train)
    # doc_pred = clf.predict(x_test)
    # print("平均值：", np.mean(doc_pred == y_test))
    # 可用 clf.score 代替以上均值
    
    score = clf.score(x_test, y_test)
    print("score:",score)

    # 准确率  召回率
    precision, recall, thresholds = precision_recall_curve(
        y_test, clf.predict(x_test))

    answer = clf.predict_proba(x_test)[:, 1]
    report = answer > 0.5
    print(classification_report(y_test, report, target_names=['net', 'pos']))

    # 特征名称
    # print(count_vec.get_feature_names())
    # 保存模型
    model_path =  "./models/clf_bayes.model"
    joblib.dump(clf, model_path, compress=0)


if __name__ == '__main__':
    main()
