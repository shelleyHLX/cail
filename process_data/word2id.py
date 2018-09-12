# coding: utf-8

from multiprocessing import Pool
import pickle
import numpy as np

save_path = '../data/'
#
file_word2id_te = 'word2id_te.npy'
file_word2id_tr = 'word2id_tr.npy'
file_word2id_va = 'word2id_va.npy'

# r
file_fact2word_te = 'fact2word_te.npy'
file_fact2word_tr = 'fact2word_tr.npy'
file_fact2word_va = 'fact2word_va.npy'

embedding_size = 256
embedding_file_name = save_path + 'sr_word2id_' + str(256) + '.pkl'
with open(embedding_file_name, 'rb') as inp:
    sr_word2id = pickle.load(inp)
dict_word2id = dict()

for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]

def get_idword(word):
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]

def get_id4words(words):
    ids = list(map(get_idword, words))
    return ids


def word2id_fun(file_fact2word, file_word2id):
    fact2word = np.load(save_path + file_fact2word)
    p = Pool()
    print(fact2word[0])
    word2id = np.asarray(list(p.map(get_id4words, fact2word)))
    print(word2id[0])
    np.save(save_path + file_word2id, word2id)
    print('save to: ', save_path + file_word2id)

if __name__ == '__main__':
    word2id_fun(file_fact2word_te, file_word2id_te)
    word2id_fun(file_fact2word_va, file_word2id_va)
    word2id_fun(file_fact2word_tr, file_word2id_tr)