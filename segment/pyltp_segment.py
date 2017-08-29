import codecs
from pyltp import SentenceSplitter, Segmentor, Postagger, NamedEntityRecognizer
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)


LTP_DATA_DIR = 'F:\\Program\\ltp_data_v3.4.0\\'  # ltp模型目录的路径
st_text = codecs.open('data/origin/st.txt', 'r', encoding='UTF-8').read()
xz_text = codecs.open('data/xuezhong.txt', 'r', encoding='UTF-8').read()

def sentence_splitter():
    """
    分句
    """
    sentence = '你好，你觉得这个例子从哪里来的？当然还是直接复制官方文档，然后改了下这里得到的。'
    sents = SentenceSplitter.split(sentence)  # 分句
    print("\n".join(sents))


def sentence_segmentor(sentence):
    """
    分词
    """
    segmentor = Segmentor()
    segmentor.load("F:\\Program\\ltp_data_v3.4.0\\cws.model")
    print("开始分词")
    words = segmentor.segment(sentence)  # 分词
    # print(words)
    words_list = list(words)
    segmentor.release()  # 释放模型

    # 分词写入到文件
    # with open("data/segdata/xz_seg_ltp.txt", "a", encoding='UTF-8') as f:
    #     f.write(" ".join(words_list))
    
       # 分词写入到文件
    with open("data/segdata/st_seg_ltp.txt", "a", encoding='UTF-8') as f:
        f.write(" ".join(words_list))

    print("分词完成")

    return words_list


def sentence_posttagger(words_list):
    """
    词性标注
    """
    posttagger = Postagger()
    posttagger.load("F:\\Program\\ltp_data_v3.4.0\\pos.model")
    postags = posttagger.postag(words_list)
    for word, tag in zip(words_list, postags):
        print(word + '/' + tag)
    
    print("=================词性标注 End============================")
    posttagger.release()

    return postags


def ner(words, postags):
    """
    命名实体识别
    """
    recognizer = NamedEntityRecognizer() # 初始化实例
    recognizer.load('F:\\Program\\ltp_data_v3.4.0\\ner.model')  # 加载模型
    netags = recognizer.recognize(words, postags)  # 命名实体识别

    for word, ntag in zip(words, netags):
        print(word + '/' + ntag)

    print("=================命名实体识别 End============================")
    recognizer.release()  # 释放模型


def main():
    # sentence_splitter()
    # xz_words_list = sentence_segmentor(xz_text)
    st_words_list = sentence_segmentor(st_text)

    # postags = sentence_posttagger(words_list)

    # ner(words_list,postags)



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
    main()
