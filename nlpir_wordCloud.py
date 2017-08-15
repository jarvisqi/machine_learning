import codecs
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
    # 对文件分词
    nlpir.FileProcess('data/st.txt'.encode("utf-8"),
                      'data/st_seg.txt'.encode("utf-8"), False)
    pynlpir.close()


def st_wordcloud():
    words_df = pd.read_csv('data/st_seg.txt', index_col=False,names=["segment"],encoding='utf-8')
    # 停用词
    stopwords = pd.read_csv("data/stop_words.txt", index_col=False,
                            quoting=3, sep="\n", names=['stopword'], encoding='utf-8')

    print(words_df)
    words_df = words_df[~words_df.segment.isin(stopwords.stopword.tolist())]
    words_stat = words_df.groupby(by=['segment'])['segment'].agg({"count": np.size})

    words_stat = words_stat.reset_index().sort_values( by=["count"], ascending=False)
    word_frequence = {x[0]: x[1] for x in words_stat.head(10).values}

    print(word_frequence)


def st_WordCloud():
    # 生成三体词云
    in_text = codecs.open('data/st.txt', 'r', encoding='UTF-8').read()
    pynlpir.open()
    key_words = pynlpir.get_key_words(in_text, max_words=300, weighted=True)
    pynlpir.close()
    print(key_words)

    font = r'C:\Windows\Fonts\msyh.ttc'  # 指定字体，不指定会报错
    color_mask = imread("resource/ge.jpg")  # 读取背景图片

    wcloud = WordCloud(
        font_path=font,
        # 背景颜色
        background_color="white",
        # 词云形状
        mask=color_mask,
        # 允许最大词汇
        max_words=1000,
        # 最大号字体
        max_font_size=80)

    wcloud.generate_from_frequencies(dict(key_words))
    # 以下代码显示图片
    plt.imshow(wcloud)
    plt.axis("off")
    plt.show()
    wcloud.to_file("data/wcimage/三体词云_1.png")


if __name__ == '__main__':

    st_WordCloud()
    # st_segment()
    # st_wordcloud()
 
