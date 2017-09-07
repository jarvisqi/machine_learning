# -*- coding:utf-8 -*-  
import thulac
import codecs
import pandas as pd
import numpy as np


def main():
    thu1 = thulac.thulac(user_dict ='data\\origin\\dict.csv',seg_only=False)  #默认模式
    text = thu1.cut("工信处女干事每月经过下属科室都要亲口交代安装工作", text=True)  #进行一句话分词
    print(text)
    text = thu1.cut("邵教授带你深度测评在线英文语法检测网站机器帮我改作文这事儿靠谱吗", text=True)  #进行一句话分词
    print(text)

    text = thu1.cut("我们中出了一个叛徒", text=True)  #进行一句话分词
    print(text)

    text = thu1.cut("韩国OLENS奥伦斯清纯超自然巧克力225度韩国OLENSteen teen natural choco", text=True)  #进行一句话分词
    print(text)


    text = thu1.cut("Moony日本尤妮佳女宝宝拉拉裤L44片纸尿裤(9-14kg适用)", text=True)  #进行一句话分词
    print(text)

    text = thu1.cut("USAUS美澳BabyGanics甘尼克宝贝奶瓶清洁剂果蔬清洗液473ml柑橘", text=True)  #进行一句话分词
    print(text)

    text = thu1.cut("日本Moony尤妮佳婴儿湿巾纸特柔新生儿可用80枚*3包 尤妮佳湿巾纸", text=True)  #进行一句话分词
    print(text)

    text = thu1.cut("远洋壹号 unlcharm 尤妮佳 moony 纸尿裤新生儿用90枚入出生~5kg", text=True)  #进行一句话分词
    print(text)
    
    text = thu1.cut("京品兴运日本尤妮佳纸尿裤s84尤妮佳尿不湿s84尤妮佳s84尤妮佳纸尿裤s尤妮佳尿不湿s纸尿裤s尤妮佳s", text=True)  #进行一句话分词
    print(text)

if __name__ == '__main__':
    # main()

    s = 'bicycle'
    # print(s[::3])
    l = list(range(10))
    l[2:5] = [20, 30]
    del l[5:7]
    print(l)
    l[3::2] = [11, 22]
    print(l)

