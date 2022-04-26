import codecs
import re
import sys

import gensim
import jieba
import numpy as np
import pandas as pd
import scipy as sp


def load_data(inputFile):
    """读取数据，分词存储
    
    Arguments:
        inputFile {[type]} -- 输入的文件
    """

    data = pd.read_excel(inputFile, names=["Title", "Content"])
    data = data.Title + data.Content

    corpora_documents = []
    for idx, item in enumerate(data):
        segmentList = segment(item)
        documents = gensim.models.doc2vec.TaggedDocument(words=segmentList, tags=[idx])
        corpora_documents.append(documents)

    return corpora_documents


def segment(doc:str):
    """分词
    
    Arguments:
        doc {str} -- 输入文档句子
    
    Returns:
        [type] -- [description]
    """
    # 停用词
    stopwords = pd.read_csv("./data/stopwords_TUH.txt", index_col=False, quoting=3,
                            names=['stopword'],
                            sep="\n",
                            encoding='utf-8')
    stopwords = list(stopwords.stopword)

    reg_html = re.compile(r'<[^>]+>', re.S)
    reg_zh = re.compile('[a-zA-Z0-9]', re.S)
    doc = reg_html.sub('', doc)
    doc = reg_zh.sub('', doc)
    wordList = list(jieba.cut(doc))
    outStr = ''
    for word in wordList:
        if word not in stopwords:
            outStr += word
            outStr += ' '
    segmentList = outStr.split(sep=" ")

    return segmentList


def main():
    corpora_documents = load_data("./data/news.xlsx")
    model = gensim.models.Doc2Vec(min_count=2, window=8,
                                  size=80,
                                  sample=1e-3,
                                  negative=5,
                                  workers=4)
    model.build_vocab(corpora_documents)
    model.train(corpora_documents,total_examples=model.corpus_count, epochs=30)
    model.save("./models/model_doc2Vec.model")
    print('vector_size :', model.vector_size)


def test():
    
    model = gensim.models.Doc2Vec.load("./models/model_doc2Vec.model") 
    # x = model.infer_vector(["特别", "香港", "青年人" ,"为主" ,"华侨", "华人", "企业", "后裔", "年轻一代", "海归" ,"青年" ])
    # # y = model.infer_vector(["中华", "人民", "共和国" ,"国家" ,"主席", "莅临", "视察", "指导"])
    # y = model.infer_vector(["香港", "青年人" ,"为主" ,"华侨", "企业", "后裔","海归" ,"青年"  ])

    # X=np.vstack([x, y])
    # sim = 1 - sp.spatial.distance.pdist(X, metric="cosine")
    # print(sim)

    # sims = model.docvecs.most_similar([vector], topn=3)
    # print(sims)
 
    data = pd.read_excel("./data/news.xlsx", names=["Title", "Content"])
    list_vect=[]

    for idx, row in data.iterrows():
        seg_title = segment(row.Title)
        seg_content = segment(row.Content)
        vector_title = model.infer_vector(seg_title)
        vector_content = model.infer_vector(seg_content)

        list_vect.append({idx: (vector_title, vector_content)})

    np.save("vect.npy", list_vect)


def compare():
    model = gensim.models.Doc2Vec.load("./models/model_doc2Vec.model") 

    word_list = segment("陈经纬：让青年人通过“紫荆谷”融入国家经济发展 以创业带动就业")
    word_vect = model.infer_vector(word_list)
    sims = model.docvecs.most_similar([word_vect], topn=3)
    print(sims)

    # result=[]
    # list_vect = np.load("vect.npy")
    # for dictData in list_vect:
    #     for key, values in dictData.items():
    #         X = np.vstack([values[0], word_vect])
    #         sim = 1 - sp.spatial.distance.pdist(X, metric="cosine")
    #         result.append((key,sim[0]))
    # sorted_result = sorted(result, key=lambda x : (x[1]), reverse = True)
    # print(sorted_result[:5])


def docsim():
    data = pd.read_excel("./data/news.xlsx", names=["Title", "Content"])
    data = data.Title + data.Content

    corpora_documents = []
    for idx, item in enumerate(data):
        segmentList = segment(item)
        corpora_documents.append(segmentList)
    # 生成字典和向量语料  
    dictionary =gensim.corpora.Dictionary(corpora_documents)  
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]  
    print(len(corpus))
    similarity = gensim.similarities.Similarity('-Similarity-index', corpus, num_features=100, num_best=5)

    word_list = segment("陈经纬：让青年人通过“紫荆谷”融入国家经济发展 以创业带动就业")
    test_corpus = dictionary.doc2bow(word_list)
    print(similarity[test_corpus_1])


if __name__ == '__main__':
    
    # main()

    # test()

    # compare()

    docsim()
