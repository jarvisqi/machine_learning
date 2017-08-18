import codecs
import jieba
from jieba.analyse import extract_tags
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)



st_text = codecs.open('data/st.txt', 'r', encoding='UTF-8').read()
jieba.dt.add_word("章北海")
jieba.dt.add_word("黑暗森林")

def jieba_segment():
    """
    分词
    """
    word_list = jieba.cut(st_text, cut_all=False)
    
    # 分词写入到文件
    with open("data/st_seg_jb.txt", "a", encoding='UTF-8') as f:
        f.write(" ".join(word_list))

    print("分词完成")


def jieba_keywords():
    """
    关键字提取
    """
    
    key_words = extract_tags(st_text, topK=300, withWeight=True, allowPOS=())
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
    print("================去掉停用词之后================")
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
    wcloud.to_file("data/wcimage/三体词云_3.png")

if __name__ == '__main__':
    jieba_segment()

    jieba_keywords()
