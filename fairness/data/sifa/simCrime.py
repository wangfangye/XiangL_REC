# -*- coding: UTF-8 -*-
'''
@project:fairness
@author:wangfy
@time:2018/12/29 10:44
'''
import data.sifa.crime as CRIME
import jieba
import logging
import warnings
import os
from gensim.models import Word2Vec, KeyedVectors
import logging
import json
from gensim.models.word2vec import LineSentence
import numpy as np
from scipy.linalg import norm
import operator

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
STOP_PATH = r'D:\code\pycharm\fairness\data\sifa\stopword.txt'


def jieba_cut_txt(text):
    #对所有犯罪事实进行分词
    with open(STOP_PATH, encoding='utf-8') as f_stop:
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
                v += model.wv[word.strip()]
                length += 1
            except KeyError:
                print("%s not in vocabulary" % (word))
        v /= length
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

import jieba
def tokenization(trafic_fact):
    all_texts = []
    stopword = open(r'D:\code\pycharm\fairness\data\sifa\stopword.txt','r',encoding='utf-8')
    stopword_dict = [word.strip() for word in stopword.readlines()]
    for fact in trafic_fact:
        outstr = ''
        result = jieba.cut(fact)
        for word in result:
            if word not in stopword_dict:
                outstr+=word
                outstr+=' '
        all_texts.append(outstr)
    return all_texts

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    allFact, allCrimalInfo, accu_label,_ = CRIME.getData(r'D:\code\pycharm\fairness\data\raw\law\Crime_data2.json')
    print(len(allFact))
    edu, eduname, accu, accuname = CRIME.init()
    print(len(edu))


    # cut_data = []
    # f_cut = open('cut_data2.json', 'w', encoding='utf-8')
    # for fact in allFact:
    #     fact_cut = jieba_cut_txt(fact)
    #     print(len(fact_cut))
    #     cut_data.append(fact_cut)
    #     f_cut.write("".join(fact_cut) + '\n')
    # f_cut.close()
    # print('success')

    # cut_data = LineSentence('cut_data.json')
    # model = Word2Vec(cut_data, min_count=1, size=100) #min_count频数阈值，大于等于5的保留,默认为1; size神经网络 NN 层单元数，它也对应了训练算法的自由程度，默认100
    # model.save('w2v.model')
    new_model = Word2Vec.load('w2v.model')
    # similarCrimeList = [{'cos': 0, 'fact': allFact[0],  'crimalInfo': allCrimalInfo[0], 'accu': accu[accu_label[0]]}]
    # print(len(similarCrimeList))
    # for i in range(1, len(allFact)):
    #     # cos = vector_similarity(new_model, allFact[0], allFact[i])
    #     similarCrimeList.append({'cos': 0, 'fact': allFact[i], 'crimalInfo': allCrimalInfo[i], 'accu': accu[accu_label[i]]})
    # print(len(similarCrimeList))

    print(new_model.wv.__contains__('test'))
    # for i in range(0, len(allFact)):
    #     cos = vector_similarity(new_model, allFact[0], allFact[i])
    #     similarCrimeList[i]['cos']=cos
    #
    # sortlist = sorted(similarCrimeList, key = lambda k : k['cos'], reverse=True)
    # for crime in sortlist[0:20]:
    #     print(crime)

    # print(jieba_cut_txt(allFact[0]))
    # print(jieba_cut_txt(allFact[1]))
    # cos = vector_similarity(new_model, allFact[0], allFact[1])
    # print(cos)
    # print(new_model.wv.n_similarity(jieba_cut_txt(allFact[2]),jieba_cut_txt(allFact[1])))



