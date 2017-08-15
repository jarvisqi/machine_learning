import codecs
import pynlpir
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)


def showWordCloud():
    # 显示词云图片
    in_text = codecs.open('data/xuezhong.txt', 'r', encoding='UTF-8').read()
    pynlpir.open()
    key_words = pynlpir.get_key_words(in_text, max_words=300, weighted=True)
    pynlpir.close()

    font = r'C:\Windows\Fonts\msyh.ttc'  # 指定字体，不指定会报错
    color_mask = imread("resource/52233.png")  # 读取背景图片
    wcloud = WordCloud(
        font_path=font,
        # 背景颜色
        background_color="white",
        # 词云形状
        mask=color_mask,
        # 允许最大词汇
        max_words=2000,
        # 最大号字体
        max_font_size=80)  # 指定字体类型、字体大小和字体颜色

    wcloud.generate_from_frequencies(dict(key_words))

    # 以下代码显示图片
    plt.imshow(wcloud)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    showWordCloud()
