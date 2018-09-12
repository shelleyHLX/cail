# coding: utf-8

from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import os
from data_helper import pad_X200_same, train_batch

# w
batch_train_path = '../data/wd_pdQS200/train/'
batch_valid_path = '../data/wd_pdQS200/valid/'
batch_test_path = '../data/wd_pdQS200/test/'

if not os.path.exists(batch_test_path):
    os.makedirs(batch_test_path)
if not os.path.exists(batch_train_path):
    os.makedirs(batch_train_path)
if not os.path.exists(batch_valid_path):
    os.makedirs(batch_valid_path)

save_path = '../data/'

# r
file_word2id_te = 'word2id_te.npy'
file_word2id_tr = 'word2id_tr.npy'
file_word2id_va = 'word2id_va.npy'

y_file_test = 'y_te.npy'
y_file_train = 'y_tr.npy'
y_file_valid = 'y_va.npy'

batch_size = 128

def wd_get_batch(wd_fact2id_path, y_path, batch_path):
    print('loading facts and ys.',
          save_path + wd_fact2id_path,
          save_path + y_path)
    facts = np.load(save_path + wd_fact2id_path)
    y = np.load(save_path + y_path)
    p = Pool()
    X = np.asarray(list(p.map(pad_X200_same, facts)), dtype=np.int64)
    p.close()
    p.join()
    sample_num = X.shape[0]
    np.random.seed(13)
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = y[new_index]
    train_batch(X, y, batch_path, batch_size)

if __name__ == '__main__':
    wd_get_batch(file_word2id_va, y_file_valid, batch_valid_path)
    wd_get_batch(file_word2id_te, y_file_test, batch_test_path)
    wd_get_batch(file_word2id_tr, y_file_train, batch_train_path)
