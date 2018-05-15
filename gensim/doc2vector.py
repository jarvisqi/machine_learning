# -*- coding: utf-8 -*-
import codecs
import re
from os import listdir
from os.path import join

import gensim
import jieba
import numpy as np
import pandas as pd


def doc_segment():
    """分词保存
    """

    # 先把所有文档的路径存进一个 array 中，docLabels：
    data_dir = "./data/corpus"
    docLabels = [f for f in listdir(data_dir) if f.endswith('.txt')]

    data = []
    for doc in docLabels:
        try:
            ws = codecs.open(data_dir + "/" + doc).read()
            doc_words = segment(ws)
            with codecs.open("data/corpus_words/{}".format(doc), "a", encoding="UTF-8") as f:
                f.write(" ".join(doc_words))
        except:
            print(doc)

def segment(doc: str):
    """中文分词

    Arguments:
        doc {str} -- 输入文本
    Returns:
        [type] -- [description]
    """
    # 停用词
    stop_words = pd.read_csv("./data/stopwords_TUH.txt", index_col=False, quoting=3,
                             names=['stopword'],
                             sep="\n",
                             encoding='utf-8')
    stop_words = list(stop_words.stopword)

    reg_html = re.compile(r'<[^>]+>', re.S)
    doc = reg_html.sub('', doc)
    doc = re.sub('[０-９]', '', doc)
    doc = re.sub('\s', '', doc)
    word_list = list(jieba.cut(doc))
    out_str = ''
    for word in word_list:
        if word not in stop_words:
            out_str += word
            out_str += ' '
    segments = out_str.split(sep=" ")

    return segments


def train():
    """训练 Doc2Vec 模型
    """

    # 先把所有文档的路径存进一个 array中，docLabels：
    data_dir = "./data/corpus_words"
    docLabels = [f for f in listdir(data_dir) if f.endswith('.txt')]

    data = []
    for doc in docLabels:
        ws = open(data_dir + "/" + doc, 'r', encoding='UTF-8').read()
        data.append(ws)

    print(len(data))
    # 训练 Doc2Vec，并保存模型：
    sentences = LabeledLineSentence(data, docLabels)
    # an empty model
    model = gensim.models.Doc2Vec(vector_size=256, window=10, min_count=5,
                                  workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
    model.build_vocab(sentences)
    print("开始训练...")
    model.train(sentences, total_examples=model.corpus_count, epochs=12)
    
    model.save("./models/doc2vec.model")
    print("model saved")


def test_model():
    print("load model")
    model = gensim.models.Doc2Vec.load('./models/doc2vec.model')

    st1 = open('./data/courpus_test/t1.txt', 'r', encoding='UTF-8').read()
    st2 = open('./data/courpus_test/t2.txt', 'r', encoding='UTF-8').read()
    # 分词
    print("segment")
    st1 = segment(st1)
    st2 = segment(st2)
    # 转成句子向量
    vect1 = sent2vec(model, st1)
    vect2 = sent2vec(model, st2)

    cos = similarity(vect1, vect2)
    print("相似度：{:.4f}".format(cos))


def similarity(a_vect, b_vect):
    """计算两个向量余弦值
    
    Arguments:
        a_vect {[type]} -- a 向量
        b_vect {[type]} -- b 向量
    
    Returns:
        [type] -- [description]
    """

    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a*b
        a_norm += a**2
        b_norm += b**2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm*b_norm)**0.5)

    return cos


def sent2vec(model, words):
    """文本转换成向量
    
    Arguments:
        model {[type]} -- Doc2Vec 模型
        words {[type]} -- 分词后的文本
    
    Returns:
        [type] -- 向量数组
    """

    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum())



class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])


if __name__ == '__main__':
    
    # doc_segment()

    # train()

    test_model()
