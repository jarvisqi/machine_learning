import pynlpir
from ctypes import c_char_p
import jieba
from jieba.analyse import extract_tags
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import word2vec
import codecs
import time
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

# pip install python-Levenshtein==0.12.0
xz_text = codecs.open('data/xuezhong.txt', 'r', encoding='UTF-8').read()
jieba.dt.add_word("剑神")
jieba.dt.add_word("李淳罡")
jieba.dt.add_word("邓太阿")
jieba.dt.add_word("大柱国")

def word_segment():
    word_list = jieba.cut(xz_text, cut_all=False)
    # 分词写入到文件
    with open("data/xz_seg_jb.txt", "a", encoding='UTF-8') as f:
        f.write(" ".join(word_list))

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


def xz_keywords():
    """
    关键字提取
    """
    key_words = extract_tags(xz_text, topK=300, withWeight=True, allowPOS=())
    # 停用词
    stopwords = pd.read_csv("data/stop_words.txt", index_col=False,
                            quoting=3, sep="\n", names=['stopword'], encoding='utf-8')
    words = [word for word, wegiht in key_words]
    keywords_df = pd.DataFrame({'keywords': words})    

    # 去掉停用词
    keywords_df = keywords_df[~keywords_df.keywords.isin(stopwords.stopword.tolist())]

    word_freq = []
    for word in keywords_df.keywords.tolist():
        for w, k in key_words:
            if word == w:
                word_freq.append((word, k))
    print(word_freq)
    show_wordCloud(word_freq)


def show_wordCloud(word_freq):
    """
    词云显示
    """
    font = r'C:\Windows\Fonts\msyh.ttc'  # 指定字体，不指定会报错
    color_mask = imread("resource/timg.jpg")  # 读取背景图片
    wcloud = WordCloud(
        font_path=font,
        # 背景颜色
        background_color="white",
        # 词云形状
        mask=color_mask,
        # 允许最大词汇
        max_words=2000,
        # 最大号字体
        max_font_size=80)

    wcloud.generate_from_frequencies(dict(word_freq))
    # 以下代码显示图片
    plt.imshow(wcloud)
    plt.axis("off")
    plt.show()
    wcloud.to_file("data/wcimage/雪中_1.png")

if __name__ == '__main__':
    s_time = time.time()
    # 分词
    word_segment()

    xz_keywords()
    # 模型训练并保存
    # mode_training()

    # main()

    print('finished time span:', time.time() - s_time)
