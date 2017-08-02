import pynlpir
from ctypes import c_char_p
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
    nlpir.AddUserWord(c_char_p("徐骁".encode()))
    nlpir.AddUserWord(c_char_p("徐渭熊".encode()))
    nlpir.AddUserWord(c_char_p("徐北枳".encode()))
    nlpir.AddUserWord(c_char_p("白狐儿脸".encode()))
    nlpir.AddUserWord(c_char_p("轩辕青锋".encode()))
    nlpir.AddUserWord(c_char_p("姜泥".encode()))
    nlpir.AddUserWord(c_char_p("大官子".encode()))
    nlpir.AddUserWord(c_char_p("北凉".encode()))
    nlpir.AddUserWord(c_char_p("小和尚".encode()))

    nlpir.FileProcess('data/xuezhong.txt'.encode("utf-8"),
                      'data/xuezhong_seg_1.txt'.encode("utf-8"), False)
    # key_words = pynlpir.get_key_words(in_text, max_words=20, weighted=True)
    # print(key_words)


def mode_traing():

    sentences = word2vec.Text8Corpus('data/xuezhong_seg_1.txt')
    model = word2vec.Word2Vec(
        sentences, min_count=20, size=2000, window=10, workers=3)
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

    print('finished')


def main():
    # 分词
    word_Segment()
    # 模型训练并保存
    mode_traing()

    # 模型使用
    w_model = word2vec.Word2Vec.load(u"models/xue.model")
    siml = w_model.similarity(u"褚禄山", u"胖子")
    print("【褚禄山】和【胖子】相似度：", siml)


if __name__ == '__main__':

    main()