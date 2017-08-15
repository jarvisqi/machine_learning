from pyltp import SentenceSplitter, Segmentor, Postagger, NamedEntityRecognizer


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
    print(words)
    words_list = list(words)
    segmentor.release()  # 释放模型

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
    sentence="我家在昆明，我现在在北京上学。中秋节你是否会想到李白？"
    words_list = sentence_segmentor(sentence)
    postags = sentence_posttagger(words_list)

    ner(words_list,postags)


if __name__ == '__main__':
    main()
