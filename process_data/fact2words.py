# coding: utf-8

import codecs
import re
import pickle
import numpy as np
import jieba

jieba.load_userdict('newdict.txt')

# w
save_path = '../data/'
file_fact2word_te = 'fact2word_te.npy'
file_fact2word_tr = 'fact2word_tr.npy'
file_fact2word_va = 'fact2word_va.npy'

# r
file_fact_test_dict = 'idfact_dict_te.pkl'
file_fact_train_dict = 'idfact_dict_tr.pkl'
file_fact_valid_dict = 'idfact_dict_va.pkl'

stopwords = []
stopwords_line = codecs.open('stopwords.txt', 'r', 'utf-8').readlines()
for stop in stopwords_line:
    stopwords.append(stop.replace('\r', '').replace('\n', ''))

stop_name = codecs.open('newdict.txt', 'r', 'utf-8').readlines()
for name in stop_name:
    stopwords.append(name.replace('\n', '').replace('\r', ''))

def valid_test_cut_words2id(file_fact_dict, file_fact_word):
    with open(save_path + file_fact_dict, 'rb') as inp:
        idfact_dict = pickle.load(inp)
    length = len(idfact_dict)
    facts_list = []
    for i in range(length):
        if i % 1000 == 0:
            print(i)
        fact_list = []
        fact_str = idfact_dict[i]

        fact_str = re.sub(r'\r', '', fact_str)
        fact_str = re.sub(r'\t', '', fact_str)
        fact_str = re.sub(r'\n', '', fact_str)
        fact_str = re.sub(r'([0-9]{4}年)?[0-9]{1,2}月([0-9]{1,2}日)?', '', fact_str)
        fact_str = re.sub(r'[0-9]{1,2}时([0-9]{1,2}分)?许?', '', fact_str)

        cut_list = jieba.cut(fact_str)
        for w in cut_list:
            if w in stopwords:
                continue
            elif '省' in w:
                continue
            elif '市' in w:
                continue
            elif '镇' in w:
                continue
            elif '村' in w:
                continue
            elif '路' in w:
                continue
            elif '县' in w:
                continue
            elif '区' in w:
                continue
            elif '城' in w:
                continue
            elif '府' in w:
                continue
            elif '庄' in w:
                continue
            elif '道' in w:
                continue
            elif '车' in w:
                continue
            elif '店' in w:
                continue
            elif '某' in w:
                continue
            elif '辆' in w:
                continue
            elif '房' in w:
                continue
            elif '馆' in w:
                continue
            elif '场' in w:
                continue
            elif '街' in w:
                continue
            elif '墙' in w:
                continue
            elif '牌' in w:
                continue
            else:
                fact_list.append(w)
        facts_list.append(fact_list)
    print(facts_list[0])
    np.save(save_path + file_fact_word, facts_list)
    print('save to: ', save_path + file_fact_word)

if __name__ == '__main__':
    valid_test_cut_words2id(file_fact_test_dict, file_fact2word_te)
    valid_test_cut_words2id(file_fact_valid_dict, file_fact2word_va)
    valid_test_cut_words2id(file_fact_train_dict, file_fact2word_tr)

