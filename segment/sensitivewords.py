# -*- coding: utf-8 -*-

import codecs
import string
import re
import numpy as np
import pandas as pd
import jieba
from sklearn.datasets import load_files  
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve,classification_report
from sklearn.model_selection import train_test_split 


st_keyword = codecs.open('./data/origin/words.txt',
                        'r', encoding='UTF-8').read()
st_text = codecs.open('./data/origin/st.txt', 'r', encoding='UTF-8').read()
stop_wordlist = codecs.open('./data/origin/stop_words.txt',
                        'r', encoding='UTF-8').read()

def main():
    r='[’，。！!”“．、"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    df_data=pd.read_csv("./data/origin/alldata.txt",sep='\t',header=None,names=['id','word','ner'])

    keywords = st_keyword.split(sep="|")
    word_data = df_data['word'][:150000]
    word_target = [1 if xt in keywords else 0 for xt in df_data['word']][:150000]

    tf_vect = TfidfVectorizer(binary = False, decode_error = 'ignore', stop_words = None)
    x_train, x_test, y_train, y_test = train_test_split(word_data, word_target, test_size = 0.2)  
    x_train = tf_vect.fit_transform(x_train)  
    x_test  = tf_vect.transform(x_test)
    # 训练
    clf = MultinomialNB().fit(x_train, y_train)  

    doc_class_predicted = clf.predict(x_test)
    print(doc_class_predicted)  

    score = clf.score(x_test, y_test)
    print("score:",score)

    # #准确率与召回率  
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))  
    answer = clf.predict_proba(x_test)[:,1]  
    report = answer > 0.5  
    print(classification_report(y_test, report, target_names = ['neg', 'pos']))  

    # x_data=["刘备","在","中南海","赌博"]
    # pred = clf.predict_proba(x_data)[:,1]

    # for index in pred:
    #     label_list.append(training_data.target_names[index])

    # print(answer)

if __name__ == '__main__':
    main()
   
   
