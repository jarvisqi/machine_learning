import pynlpir
from ctypes import c_char_p
from pynlpir import nlpir

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim.models import word2vec
import codecs
import time


# class WordSegment(object):
#     def __init__(self):
#         if not nlpir.Init(nlpir.PACKAGE_DIR, nlpir.UTF8_CODE, None):
#             print('Initialize NLPIR failed')
#             exit(-11111)


def word_segment():
    in_text = codecs.open('data/xuezhong.txt', 'r', encoding='UTF-8').read()
    pynlpir.open()

    # 添加自定义字典
    nlpir.AddUserWord(c_char_p("徐骁".encode()))
    nlpir.AddUserWord(c_char_p("老怪物".encode()))
    nlpir.AddUserWord(c_char_p("徐渭熊".encode()))
    nlpir.AddUserWord(c_char_p("徐北枳".encode()))
    nlpir.AddUserWord(c_char_p("白狐儿脸".encode()))
    nlpir.AddUserWord(c_char_p("轩辕青锋".encode()))
    nlpir.AddUserWord(c_char_p("姜泥".encode()))
    nlpir.AddUserWord(c_char_p("大官子".encode()))
    nlpir.AddUserWord(c_char_p("北凉".encode()))
    nlpir.AddUserWord(c_char_p("小和尚".encode()))

    # 对文件分词
    nlpir.FileProcess('data/xuezhong.txt'.encode("utf-8"),
                      'data/xuezhong_seg_1.txt'.encode("utf-8"), False)

    # key_words = pynlpir.get_key_words(in_text, max_words=100, weighted=True)
    pynlpir.close()
    print(key_words)

    print("segment finished")


def mode_training():
    """
    模型训练
    """
    # 读取文件下下面的文件
    # sentences = MySentences('/some/directory')
    # 分词数据
    sentences = word2vec.Text8Corpus('data/xuezhong_seg_1.txt')
    # 训练 size参数主要是用来设置神经网络的层数
    # workers参数用于设置并发训练时候的线程数，不过仅当Cython安装的情况
    model = word2vec.Word2Vec(
        sentences, min_count=20, size=4000, window=10, workers=4)


    # model.sort_vocab()

    # 计算两个词的相似度/相关程度
    # simil_1 = model.wv.similarity(u"王仙芝", u"老怪物")
    # simil_2 = model.wv.similarity(u"徐凤年", u"殿下")
    # print("【王仙芝】和【老怪物】相似度：", simil_1)
    # print("【徐凤年】和【世子】相似度：", simil_2)

    # # 计算某个词的相关词列表
    # lar = model.wv.most_similar(u"徐凤年", topn=20)  # 20个最相关的
    # print("【徐凤年】相关性：", lar)

    # 保存模型，以便重用
    model.save(u"models/xue.model")
    print("training finished")


def main():
    # 模型使用
    w_model = word2vec.Word2Vec.load(u"models/xue.model")
    # siml = w_model.similarity(u"褚禄山", u"胖子")
    sim2 = w_model.wv.similarity(u"大官子", u"曹长卿")
    # print("【褚禄山】和【胖子】相似度：", siml)
    print("【大官子】和【曹长卿】相似度：", sim2)


if __name__ == '__main__':
    s_time = time.time()
    # 分词
    # word_segment()
    # 模型训练并保存
    # mode_training()

    main()

    print('finished time span:', time.time() - s_time)
