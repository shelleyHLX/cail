# coding: utf-8

import numpy as np
import os

def pad_X300(words, max_len=300):
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[:max_len]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])

def pad_X200_0(words, max_len=200):
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[(words_len - max_len):]
    return np.hstack([np.zeros(max_len-words_len, dtype=int), words])

def pad_X200_same(words, max_len=200):
    words_len = len(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[(words_len - max_len):]
    if words_len < max_len:
        num = int(max_len / words_len)
        words_num = words * num
        words_num = np.asarray(words_num)
        end_idx = max_len - words_len * num - 1
        words_last = words[-1:-end_idx-2:-1]
        words_last = np.asarray(words_last)
        return np.hstack([words_last, words_num])

def to_categorical(topics):
    n_sample = len(topics)
    y = np.zeros(shape=(n_sample, 202))

    for i in range(n_sample):
        topic_index = topics[i]
        y[i, topic_index] = 1
    return y

def train_batch(X, y, batch_path, batch_size=128):
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    batch_num = 0
    for start in list(range(0, sample_num, batch_size)):
        print(batch_num)
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npz'
        X_batch = X[start:end]
        y_batch = y[start:end]
        np.savez(batch_name, X=X_batch, y=y_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num + 1))

