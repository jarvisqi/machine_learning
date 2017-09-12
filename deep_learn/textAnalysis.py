# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import jieba
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential,model_from_yaml, load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
np.random.seed(7)
import sys
sys.setrecursionlimit(1000000)

# set parameters:
vocab_dim = 128
maxlen = 120
window_size = 7
batch_size = 256
cpu_count = multiprocessing.cpu_count()


def loadfile():
    """
    加载数据
    """
    neg = pd.read_excel("./data/text/neg.xls", header=None, index=None)
    pos = pd.read_excel("./data/text/pos.xls", header=None, index=None)

    # # 分词
    # def cw(x): return list(jieba.cut(x))
    # neg['words'] = neg[0].apply(cw)
    # pos['words'] = pos[0].apply(cw)

    # # 数据合并
    # y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    # data = np.concatenate((neg['words'], pos['words']))
    # # 切分训练数据
    # x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

    # np.save('svm_data/y_train.npy', y_train)
    # np.save('svm_data/y_test.npy', y_test)

    # return x_train, x_test

    combined = np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int),
                        np.zeros(len(neg), dtype=int)))
    return combined, y


def create_dictionaries(model=None, combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word]
                 for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined = parse_dataset(combined)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(text):

    model = word2vec.Word2Vec(size=vocab_dim, min_count=10, window=window_size, workers=cpu_count,
                              iter=1)
    model.build_vocab(text)
    model.train(text, total_examples=model.corpus_count, epochs=model.iter)
    model.save('./data/text/Word2vec_model.model')
    index_dict, word_vectors, text = create_dictionaries(
        model=model, combined=text)
    return index_dict, word_vectors, text


def get_data(index_dict, word_vectors, inputTexts, y):

    input_dim = len(index_dict) + 1              # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    x_train, x_test, y_train, y_test = train_test_split(inputTexts, y, test_size=0.1)
    print(x_train.shape, y_train.shape)

    return input_dim, x_train, y_train, x_test, y_test


# 定义网络结构
def train_lstm(input_dim, x_train, y_train, x_test, y_test):
    print('设计模型 Model...')

    model = Sequential()

    # model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True,
    #                     weights=[embedding_weights], input_length=input_length))  # Adding Input Length
    # model.add(LSTM(units=50,activation="sigmoid",recurrent_activation="hard_sigmoid"))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    print(input_dim)
    model.add(Embedding(input_dim, vocab_dim, mask_zero=True, input_length=maxlen))
    model.add(LSTM(128, activation="sigmoid",dropout=0.25,recurrent_dropout=0.25))
    model.add(Dense(64,activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(32,activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(16,activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(1,activation="sigmoid"))


    # Test score: 0.263636458462
    # Test accuracy: 0.916153483767

    print('编译模型...')   # 使用 adam优化
    sgd = Adam(lr=0.003)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print("训练...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=10,verbose=1, validation_data=(x_test, y_test))

    print("评估...")
    score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    yaml_string = model.to_yaml()
    with open('./data/text/lstm.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./data/text/lstm.h5')


def train():
    """
    训练模型，并保存

    """
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('分词...')
    combined = [jieba.lcut(document.replace('\n', ''))for document in combined]
    print('训练 Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    n_symbols, x_train, y_train, x_test, y_test = get_data(
        index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)

    train_lstm(n_symbols, x_train, y_train, x_test, y_test)


def predictData():
    """
    使用模型预测真实数据

    """
    input_texts = ["价格不便宜，花了190元。", "垃圾", "东西很好，超值",
                   "服务态度好", "差评，不要买", "发货速度超级快！", "一般吧， 价格还不便宜"]

    texts = [jieba.lcut(document.replace('\n', '')) for document in input_texts]
    word_model = word2vec.Word2Vec.load('./data/text/Word2vec_model.model')
    w2indx, w2vec, texts = create_dictionaries(word_model, texts)
    # 加载网络结构
    with open('./data/text/lstm.yaml', 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
    model = model_from_yaml(loaded_model_yaml)
    # 加载模型权重
    model.load_weights("./data/text/lstm.h5")
    print("model Loaded")

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # 预测
    pred_result = model.predict_classes(texts)
    labels = [int(round(x[0])) for x in pred_result]
    label2word = {1: '正面', 0: '负面'}
    for i in range(len(pred_result)):
        print('{} -------- {}'.format(label2word[labels[i]], input_texts[i]))


if __name__ == '__main__':
    
    # train()

    predictData()
