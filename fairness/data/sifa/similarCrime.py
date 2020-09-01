# encoding: utf-8
'''
@Author: shuhan Wei
@File: similarCrime.py
@Time: 2018/12/26 23:51
'''

import jieba
import crime
import logging
import os
from gensim.models import Word2Vec, KeyedVectors
import logging
import json
from gensim.models.word2vec import LineSentence
import numpy as np
from scipy.linalg import norm
import operator

allFact, allCrimalInfo, accu_label = crime.getData('./data/twoCrimeData2.json')
edu, eduname, accu, accuname = crime.init()

def jieba_cut_txt(text):
    #对所有犯罪事实进行分词
    with open('./data/stopwords.txt', encoding='utf-8') as f_stop:
        f_stop_text = f_stop.read()
        f_stop_seg_list = f_stop_text.splitlines()
        mywordlist = []
        seg_list = jieba.cut(text, cut_all=False)
        liststr = "/ ".join(seg_list)
        for myword in liststr.split('/'):
            if not (myword.strip() in f_stop_seg_list) and len(myword.strip()) > 1:
                mywordlist.append(myword)
        f_stop.close()
        return mywordlist


def vector_similarity(model,s1, s2):
    def sentence_vector(s):
        words = jieba_cut_txt(s)
        v = np.zeros(100)
        length = 0
        for word in words:
            try:
                v += model[word.strip()]
                length += 1
            except KeyError:
                print("%s not in vocabulary" % (word))
        v /= length
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # cut_data = []
    # f_cut = open('./data/cut_data.json', 'w', encoding='utf-8')
    # for fact in allFact:
    #     fact_cut = jieba_cut_txt(fact)
    #     cut_data.append(fact_cut)
    #     f_cut.write("".join(fact_cut) + '\n')
    # f_cut.close()

    # cut_data = LineSentence('./data/cut_data.json')
    # model = Word2Vec(cut_data, min_count=3, size=100) #min_count频数阈值，大于等于5的保留,默认为1; size神经网络 NN 层单元数，它也对应了训练算法的自由程度，默认100
    # model.save('./model/w2v.model')
    new_model = Word2Vec.load('./model/w2v.model')
    similarCrimeList = [{'cos': 1, 'fact': allFact[0],  'crimalInfo': allCrimalInfo[0], 'accu': accu[accu_label[0]]}]

    for i in range(1, len(allFact)):
        cos = vector_similarity(new_model, allFact[0], allFact[i])
        similarCrimeList.append({'cos': cos, 'fact': allFact[i], 'crimalInfo': allCrimalInfo[i], 'accu': accu[accu_label[i]]})

    similarCrimeList = sorted(similarCrimeList, key = lambda k : k['cos'], reverse=True)
    for crime in similarCrimeList[0:20]:
        print(crime)

