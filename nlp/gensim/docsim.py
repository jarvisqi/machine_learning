# -*- coding: UTF-8 -*-
import re
import sys

import jieba
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities

sys.path.append("utility")
from utility.async_decor import async_wrapper
from utility.logger_decor import create_logger, exception


class Docsim(object):
    """
    文本相似度
    """
    logger = create_logger()

    def __init__(self, keep_val: float = 0.8):
        """
        :param keep_val: 设定的阈值
        """
        self.keep_val = keep_val

    def load_data(self, input_file):
        """读取数据，分词存储

        Arguments:
            inputFile {[type]} -- 输入的文件
        """

        df_data = pd.read_excel(input_file, names=["SysNo", "Title", "Content"])
        corpora_documents = []
        idx_dic = {}
        for idx, row in df_data.iterrows():
            seg_title = self.segment(row.Title)
            seg_content = self.segment(row.Content)
            idx_dic[idx] = row.SysNo
            corpora_documents.append((seg_title, seg_content))

        titles = [t for t, _ in corpora_documents]
        contents = [c for _, c in corpora_documents]
        # 保存分词文件
        np.save("./data/title_words.npy", titles)
        np.save("./data/content_words.npy", contents)
        # 保存索引字典
        with open('./data/idx_dic.dic', 'w') as f:
            f.write(str(idx_dic))

        return corpora_documents

    def segment(self, doc: str):
        """分词

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
        word_list = list(jieba.cut(doc))
        out_str = ''
        for word in word_list:
            if word not in stop_words:
                out_str += word
                out_str += ' '
        segments = out_str.split(sep=" ")

        return segments

    def train(self, prefix: str, corporas: list):
        """ 训练模型
        保存字典，语料，模型到磁盘

        Arguments:
            prefix {str} -- 模型名称前缀
            corpora_documents {list} -- 分词后的文本
        """
        # 生成字典和向量语料
        dictionary = corpora.Dictionary(corporas)
        dictionary.save('./models/{}_dict.dic'.format(prefix))  # 保存生成的词典

        corpus = [dictionary.doc2bow(text) for text in corporas]
        corpora.MmCorpus.serialize('./models/{}_corpuse.mm'.format(prefix), corpus)  # 保存生成的语料
        tfidf_model = models.TfidfModel(corpus)
        tfidf_model.save("./models/{}_tfidf_model.model".format(prefix))  # 保存Tfidf模型

    @exception(logger)
    def calc_similarity(self, prefix: str, text: str):
        """计算相似度
        返回索引和余弦值

        Arguments:
            prefix {str} -- 模型前缀
            text {str} -- 文本数据
            value {float} -- 设定的阈值，返回大于这个值的数据
        """
        dictionary = corpora.Dictionary.load('./models/{}_dict.dic'.format(prefix))  # 加载字典
        corpus = corpora.MmCorpus('./models/{}_corpuse.mm'.format(prefix))  # 加载语料
        tfidf_model = models.TfidfModel.load("./models/{}_tfidf_model.model".format(prefix))  # 加载Tfidf模型
        corpus_tfidf = tfidf_model[corpus]

        lsi = models.LsiModel(corpus_tfidf)
        corpus_lsi = lsi[corpus_tfidf]
        similarity_lsi = similarities.Similarity('./models/similarity-lsi-index',
                                                 corpus_lsi,
                                                 num_features=400,
                                                 num_best=3)
        cut_raw = self.segment(text)  # 1.分词
        corpus = dictionary.doc2bow(cut_raw)  # 2.转换成bow向量
        corpus_tfidf = tfidf_model[corpus]  # 3.计算tfidf值
        corpus_lsi = lsi[corpus_tfidf]  # 4.计算lsi值
        sims = similarity_lsi[corpus_lsi]

        with open('./data/idx_dic.dic', 'r') as f:
            dt = f.read()
            idx_dic = eval(dt)

        result = []
        if sims is not None:
            result = [idx_dic[idx] for idx, val in sims if val > self.keep_val]

        return result

    @async_wrapper
    def update_model(self, prefix: str, sysno: int, doc: str):
        """
        更新字典
        :param prefix:
        :param sysno: 系统编号
        :param doc:   文本内容
        :return:
        """

        corporas = self.segment(doc)
        # # 更新字典
        dictionary = corpora.Dictionary.load('./models/{}_dict.dic'.format(prefix))  # 加载
        dictionary.add_documents([corporas])
        dictionary.save('./models/{}_dict.dic'.format(prefix))  # 保存生成的词典

        corporas_docs = np.load("./data/{}_words.npy".format(prefix))
        corporas_docs = list(corporas_docs)
        corporas_docs.append(corporas)
        np.save("./data/{}_words.npy".format(prefix), corporas_docs)
        # 更新corpus
        corpus = [dictionary.doc2bow(text) for text in corporas_docs]
        corpora.MmCorpus.serialize('./models/{}_corpuse.mm'.format(prefix), corpus)

        # 更新TfidfModel
        tfidf_model = models.TfidfModel(corpus)
        tfidf_model.save("./models/{}_tfidf_model.model".format(prefix))

        # 更新索引字典
        with open('./data/idx_dic.dic', 'r') as f:
            dt = f.read()
            idx_dic = eval(dt)

        if sysno not in idx_dic.values():
            idx_dic[len(idx_dic)] = sysno

        with open('./data/idx_dic.dic', 'w') as f:
            f.write(str(idx_dic))


if __name__ == '__main__':
    doc = Docsim()
    documents = doc.load_data("./data/news.xlsx")

    title = [t for t, _ in documents]
    doc.train("title", title)

    content = [c for _, c in documents]
    doc.train("content", content)

    # ids = doc.get_simdoc("title", "分布式系统中的多台计算机之间在空间位置上可以随意分布")
    # ids = doc.get_simdoc("title", "字典是另一种可变容器模型且可存储任意类型对象")
    # print(ids)

    # doc.update_model("title",301, "分布式系统中的多台计算机之间在空间位置上可以随意分布")
    # doc.update_model("title", 302, "字典是另一种可变容器模型且可存储任意类型对象")

    # doc.get_simdoc("title", "分布式系统中的多台计算机之间在空间位置上可以随意分布")
    # doc.update_model("title", 302, "字典是另一种可变容器模型且可存储任意类型对象")
