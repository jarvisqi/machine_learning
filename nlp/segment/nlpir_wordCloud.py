# -*- coding: utf-8 -*-

import codecs
from ctypes import c_char_p
import pandas as pd
import numpy as np
import pynlpir
from pynlpir import nlpir
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

# 显示词云图片


def st_segment():
    # 分词
    pynlpir.open()
    # 添加自定义字典
    nlpir.AddUserWord(c_char_p("三体".encode()))
    nlpir.AddUserWord(c_char_p("罗辑".encode()))
    # 对文件分词
    nlpir.FileProcess('data/st.txt'.encode("utf-8"),
                      'data/segdata/st_seg.txt'.encode("utf-8"), False)
    pynlpir.close()


def st_wordcloud():
    words_df = pd.read_csv('data/segdata/st_seg.txt', index_col=False,
                           names=["segment"], encoding='utf-8')
    # 停用词
    stopwords = pd.read_csv("data/stop_words.txt", index_col=False,
                            quoting=3, sep="\n", names=['stopword'], encoding='utf-8')

    print(words_df)
    words_df = words_df[~words_df.segment.isin(stopwords.stopword.tolist())]
    words_stat = words_df.groupby(by=['segment'])[
        'segment'].agg({"count": np.size})

    words_stat = words_stat.reset_index().sort_values(
        by=["count"], ascending=False)
    word_frequence = {x[0]: x[1] for x in words_stat.head(10).values}

    print(word_frequence)


def st_WordCloud():
    # 生成三体词云
    in_text = codecs.open('data/st.txt', 'r', encoding='UTF-8').read()
    pynlpir.open()

    nlpir.AddUserWord(c_char_p("三体".encode()))
    nlpir.AddUserWord(c_char_p("罗辑".encode()))
    key_words = pynlpir.get_key_words(in_text, max_words=300, weighted=True)
    # 停用词
    stopwords = pd.read_csv("data/stop_words.txt", index_col=False,
                            quoting=3, sep="\n", names=['stopword'], encoding='utf-8')
    words = [word for word, wegiht in key_words]
    keywords_df = pd.DataFrame({'keywords': words})
    # 去掉停用词
    keywords_df = keywords_df[~keywords_df.keywords.isin(
        stopwords.stopword.tolist())]

    word_freq = []
    for word in keywords_df.keywords.tolist():
        for w, k in key_words:
            if word == w:
                word_freq.append((word, k))

    pynlpir.close()
    print(word_freq)

    font = r'C:\Windows\Fonts\msyh.ttc'  # 指定字体，不指定会报错
    # color_mask = imread("resource/ge.jpg")  # 读取背景图片
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
    wcloud.to_file("data/wcimage/三体词云_2.png")


if __name__ == '__main__':

    st_WordCloud()
    # st_segment()
    # st_wordcloud()
