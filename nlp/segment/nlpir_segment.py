# -*- coding: utf-8 -*-

import pynlpir
from ctypes import c_char_p
import codecs


def main():
    pynlpir.open()
    # pynlpir.nlpir.AddUserWord(c_char_p("手机壳".encode()))
    # pynlpir.nlpir.AddUserWord(c_char_p("炫亮".encode()))
    # text = '弗洛米iPhone7/7plus手机壳/保护套苹果7plus超薄全包硅胶透明电镀软壳5.5英寸炫亮黑☆炫亮电镀'
    # text="“赶考”五年成绩非凡 全面从严治党永远在路上"
    text = codecs.open('data/new.txt', 'r', encoding='UTF-8').read()
    r_out = pynlpir.segment(text, pos_english=False)
    key_words = pynlpir.get_key_words(text, weighted=True)

    pynlpir.close()
    print(key_words)

    for x in r_out:
        print(x)

    # print(key_words)


if __name__ == '__main__':
    main()
