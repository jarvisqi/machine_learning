import codecs
import sys
import re
import jieba
import pandas as pd
import gensim


def load_data(inputFile, outputFile):
    """读取数据，分词存储
    
    Arguments:
        inputFile {[type]} -- 输入的文件
        outputFile {[type]} -- 输出的文件
    """

    data = pd.read_excel(inputFile, names=["Title", "Content"])
    data = data.Title + data.Content

    # 停用词
    stopwords = pd.read_csv("./data/stopwords_TUH.txt", index_col=False, quoting=3,
                            names=['stopword'],
                            sep="\n",
                            encoding='utf-8')
    stopwords = list(stopwords.stopword)
    corpora_documents = []
    for item in data:
        reg = re.compile(r'<[^>]+>', re.S)
        item = reg.sub('', item)
        wordList = list(jieba.cut(item))
        outStr = ''
        segment = []
        for word in wordList:
            if word.strip() not in stopwords:
                outStr += word
                outStr += ' '
        segmentList = outStr.split(sep=" ")
        corpora_documents.append(segmentList)
        with open("./data/segment.txt", "a", encoding='UTF-8') as f:
            f.write(" ".join(segmentList))

    return corpora_documents


def main():
    corpora_documents = load_data("./data/news.xlsx", "./data/segment.txt")
    dictionary = gensim.corpora.Dictionary(documents=corpora_documents)
    print(dictionary)
    dictionary.save('./data/news_dict.txt')                  # 保存生成的词典
    dictionary = gensim.corpora.Dictionary.load('./data/news_dict.txt')     # 加载

    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    tfidf_model = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    tfidf_model.save("./data/news_tf.tfidf")
    tfidf = gensim.models.TfidfModel.load("./data/news_tf.tfidf")


    # data = codecs.open('data/xuezhong.txt', 'r', encoding='UTF-8').read()
    # model = gensim.models.KeyedVectors.load_word2vec_format("./models/embedding_64.bin", binary=True)
    # similar = model.most_similar('向量', topn=3)
    # print(similar)
    # vec = model["计算机"]
    # print(vec,type(vec))
    # print(sys.getsizeof(vec))

    # texts = ["用于", "计算", "长", "文本", "向量", "工具"]
    # sentence = gensim.models.word2vec.Word2Vec(sentence=texts, size=64, window=5, min_count=5, workers=4)
    pass


if __name__ == '__main__':
    main()

    # load_data("./data/news.xlsx", "./data/segment.txt")
