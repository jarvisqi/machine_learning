import pynlpir
from ctypes import c_char_p


def main():
    pynlpir.open()
    pynlpir.nlpir.AddUserWord(c_char_p("手机壳".encode()))
    pynlpir.nlpir.AddUserWord(c_char_p("炫亮".encode()))
    text = '弗洛米iPhone7/7plus手机壳/保护套苹果7plus超薄全包硅胶透明电镀软壳5.5英寸炫亮黑☆炫亮电镀'
    r_out = pynlpir.segment(text, pos_english=False)
    key_words = pynlpir.get_key_words(text, weighted=True)
    pynlpir.close()

    for x in r_out:
        print(x)

    # print(key_words)


if __name__ == '__main__':
    main()
