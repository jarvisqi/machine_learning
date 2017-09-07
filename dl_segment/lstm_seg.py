# -*- coding:utf-8 -*-
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Sequential, Model,load_model
from keras.models import load_model



s = open('data/training/msr_train.txt').read()
s = s.split('\r\n')
maxlen = 32


def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

data = []  # 生成训练样本
labels = []

def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        labels.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = labels
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})

chars = []  # 统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars) + 1)


#生成适合模型输入的格式
from keras.utils import np_utils
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
d['y'] = d['label'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5), tag[x].values.reshape((-1,1))))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))

#设计模型
word_size = 128
maxlen = 32

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




# print("training start")
# batch_size = 1024
# hist = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)
# print("training end")
# print(hist.history)  # model.fit在运行结束后返回一个History对象，其中含有的history属性包含了训练过程中损失函数的值以及其他度量指标。
# model.save("data/models/lstm_seg.h5")


# 利用 labels（即状态序列）来统计转移概率
# 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
# A统计状态转移的频数
A = {
    'sb': 0.3,
    'ss': 0.3,
    'be': 0.3,
    'bm': 0.3,
    'me': 0.3,
    'mm': 0.3,
    'eb': 0.3,
    'es': 0.3
}
# zy 表示转移概率矩阵
zy = dict()
for label in labels:
    for t in range(len(label) - 1):
        key = label[t] + label[t + 1]
        A[key] += 1.0

zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
zy['ss'] = 1.0 - zy['sb']
zy['be'] = A['be'] / (A['be'] + A['bm'])
zy['bm'] = 1.0 - zy['be']
zy['me'] = A['me'] / (A['me'] + A['mm'])
zy['mm'] = 1.0 - zy['me']
zy['eb'] = A['eb'] / (A['eb'] + A['es'])
zy['es'] = 1.0 - zy['eb']
keys = sorted(zy.keys())
zy = {i: np.log(zy[i]) for i in zy.keys()}


def viterbi(nodes):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + \
                        nodes[layer][node_now] + zy[path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]   # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value

    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values() 
    return sr_paths.index[-1]  


def simple_cut(strtext):
    if strtext:
        model = load_model("data/models/lstm_seg.h5")
        x_test = [list(chars[list(strtext)].fillna(0).astype(int)) + [0] * (maxlen - len(strtext))]
        r = model.predict(np.array(x_test), verbose=False)[0][:len(strtext)]
        r = np.log(r)
        nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
        tags = viterbi(nodes)
        words = []
        for i in range(len(strtext)):
            if tags[i] in ['s', 'b']:
                words.append(strtext[i])
            else:
                words[-1] += strtext[i]
        return words
    else:
        return []


not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')

def cut_word(strtext):
    result = []
    j = 0
    for i in not_cuts.finditer(strtext):
        result.extend(simple_cut(strtext[j:i.start()]))
        result.append(strtext[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(strtext[j:]))
    return result


def main():
    # strtext="人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词"
    strtext="京品兴运日本尤妮佳纸尿裤s84尤妮佳尿不湿s84尤妮佳s84尤妮佳纸尿裤s尤妮佳尿不湿s纸尿裤s尤妮佳s"
    result = cut_word(strtext)
    rss = ''
    for each in result:
        rss = rss + each + ' / '
    print(rss)


if __name__ == '__main__':
    main()
