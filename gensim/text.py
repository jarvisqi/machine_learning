import jieba
import pandas as pd
from gensim import corpora, models, similarities

# 训练样本
raw_documents = [
    '0无偿居间介绍买卖毒品的行为应如何定性',
    '1吸毒男动态持有大量毒品的行为该如何认定',
    '2如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
    '3为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
    '4将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
    '5为获报酬帮人购买毒品的行为该如何认定',
    '6毒贩出狱后再次够买毒品途中被抓的行为认定',
    '7虚夸毒品功效劝人吸食毒品的行为该如何认定',
    '8妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
    '9一方未签字办理的结婚登记是否有效',
    '10夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
    '11结婚前对方父母出资购买的住房写我们二人的名字有效吗',
    '12身份证被别人冒用无法登记结婚怎么办？',
    '13同居后又与他人登记结婚是否构成重婚罪',
    '14未办登记只举办结婚仪式可起诉离婚吗',
    '15同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
]

def main():
    corpora_documents = []
    for item_text in raw_documents:
        item_str = list(jieba.cut(item_text))
        corpora_documents.append(item_str)

    dictionary = corpora.Dictionary(corpora_documents)
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]

    similarity =similarities.Similarity('-Similarity-index', corpus, num_features=400)

    test_data_1 = '你好，我想问一下我想离婚他不想离，孩子他说不要，是六个月就自动生效离婚'
    test_cut_raw_1 = jieba.cut(test_data_1)
    test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
    similarity.num_best = 5
    # 返回最相似的样本材料,(index_of_document, similarity) tuples
    print(similarity[test_corpus_1])   

if __name__ == '__main__':
    main()

