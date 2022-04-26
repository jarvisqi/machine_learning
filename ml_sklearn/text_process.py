import tarfile
from urllib import request
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_data():
    path = "./data/news.tar.gz"
    request.urlretrieve("http://jizhi-10061919.cos.myqcloud.com/sklearn/20news-bydate.tar.gz", path)
    tar = tarfile.open(path, "r:gz")
    tar.extractall(path="./data/")
    tar.close()
    print("下载解压完成")
    # 选取需要下载的新闻分类
    categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
    twenty_train = load_files("./data/20news-bydate/20news-bydate-train",
                              categories=categories, load_content=True,
                              encoding='latin1',
                              decode_error='strict',
                              shuffle=True, random_state=42 )
    return twenty_train


def text_vector():
    categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
    twenty_train = load_files("./data/20news-bydate/20news-bydate-train",
                              categories=categories, load_content=True,
                              encoding='latin1',
                              decode_error='strict',
                              shuffle=True, random_state=42 )

    print(twenty_train.data[:1])
    # 统计词语出现次数
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print("训练数据共有{0}篇，词汇数量为{1}".format(X_train_counts.shape[0], X_train_counts.shape[1]))
    count=count_vect.vocabulary_.get(u"algorithm")
    print("algorithm的出现次数为{0}".format(count))

    # 使用tf-idf方法提取文本特征
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)
    
    # 贝叶斯分类器
    clf= MultinomialNB().fit(X_train_tfidf,twenty_train.target)

    # 预测用的新字符串，你可以将其替换为任意英文句子
    docs_new = ['Nvidia is awesome!']
    # 字符串处理
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    # 进行预测
    predicted = clf.predict(X_new_tfidf)
    print(predicted)
    for doc,category in zip(docs_new,predicted):
        print("%r => %s"% (doc,twenty_train.target_names[category]))

    # 建立Pipeline
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    # 训练分类器
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
    # 打印分类器信息
    # print(text_clf)

    # 获取测试数据
    twenty_test = load_files('./data/20news-bydate/20news-bydate-test',
                             categories=categories,
                             load_content=True,
                             encoding='latin1',
                             decode_error='strict',
                             shuffle=True, random_state=42)
    docs_test = twenty_test.data
    # 使用测试数据进行分类预测
    predicted = text_clf.predict(docs_test)
    # 计算预测结果的准确率
    print("准确率为：", np.mean(predicted == twenty_test.target))

    print("打印分类只能指标：")
    report = metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names)
    print(report)
    print("打印混淆矩阵：")
    con_max = metrics.confusion_matrix(twenty_test.target, predicted)
    print(con_max)

    # 网格搜索

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3)
    }

    gs_clf=GridSearchCV(text_clf,parameters)
    print(gs_clf)

    # 使用部分训练数据训练分类器
    gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
    # 查看分类器对于新文本的预测结果，你可以自行改变下方的字符串来观察分类效果
    twenty_train.target_names[gs_clf.predict(['An apple a day keeps doctor away'])[0]]
    print("最佳准确率：%r" % (gs_clf.best_score_))
    print("参数列表：")
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


if __name__ == '__main__':

    # get_data()

    text_vector()
