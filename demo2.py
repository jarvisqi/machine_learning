import pynlpir
from pynlpir import nlpir

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim.models import word2vec
import codecs

# class WordSegment(object):
#     def __init__(self):
#         if not nlpir.Init(nlpir.PACKAGE_DIR, nlpir.UTF8_CODE, None):
#             print('Initialize NLPIR failed')
#             exit(-11111)


def word_Segment():
    in_text = codecs.open('data/xuezhong.txt', 'r', encoding='UTF-8').read()
    pynlpir.open()
    # 文件分词
    nlpir.FileProcess('data/xuezhong.txt'.encode("utf-8"),
                      'data/xuezhong_seg_1.txt'.encode("utf-8"), False)
    # key_words = pynlpir.get_key_words(in_text, max_words=20, weighted=True)
    # print(key_words)


def main():
    sentences = word2vec.Text8Corpus('data/xuezhong_seg_1.txt')
    model = word2vec.Word2Vec(
        sentences, min_count=20, size=800, window=8, workers=3)
    # 计算两个词的相似度/相关程度
    simil_1 = model.similarity(u"王仙芝", u"怪物")
    simil_2 = model.similarity(u"徐凤年", u"殿下")
    print("【王仙芝】和【怪物】相似度：", simil_1)
    print("【徐凤年】和【世子】相似度：", simil_2)

    # 计算某个词的相关词列表
    lar = model.most_similar(u"徐凤年", topn=20)  # 20个最相关的
    print("【徐凤年】相关行：", lar)

    # 保存模型，以便重用
    model.save(u"models/xue.model")

    # 加载模型
    # model_2 = word2vec.Word2Vec.load("text8.model")

    print('finished')


if __name__ == '__main__':
    main()