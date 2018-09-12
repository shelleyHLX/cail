# coding: utf-8


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import word2vec
import pickle
import os
import codecs

SPECIAL_SYMBOL = ['<PAD>', '<UNK>']

read_path = '../word2vector/'
save_path = '../data/'

def get_word_embedding(embedding_size, file_embed):
    file_embedding_name = file_embed + '_' + str(embedding_size) + '.txt'
    wv = word2vec.load(read_path + file_embedding_name)
    word_embedding = wv.vectors
    words = wv.vocab
    n_special_sym = len(SPECIAL_SYMBOL)
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    word_embedding = np.vstack([vec_special_sym, word_embedding])
    words =  SPECIAL_SYMBOL + words.tolist()
    sr_word2id = pd.Series(range(0, len(words)), index=words)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + file_embed + '_' + str(embedding_size) + '.npy', word_embedding)
    print('Saving to: ',
          save_path + file_embed + '_' + str(embedding_size) + '.npy')
    with open(save_path + 'sr_word2id' + '_' + str(embedding_size) + '.pkl', 'wb') as outp:
        pickle.dump(sr_word2id, outp)
    print('Saving to: ',
          save_path + 'sr_word2id' + '_' + str(embedding_size) + '.pkl')

    print('save dealt word embedding')
    w_file_name = save_path + file_embed + '_' + str(embedding_size) + '_dealt' + '.txt'
    w_embed = codecs.open(w_file_name, 'a', 'utf-8')
    w_embed.write(str(len(words)) + ' ' + str(embedding_size) + '\n')
    for idx in range(len(words)):
        w_embed.write(words[idx])
        for vc in range(embedding_size):
            w_embed.write(' ' + str(word_embedding[idx][vc]))
        w_embed.write('\n')
    w_embed.close()


if __name__ == '__main__':
    embedding_size = 256
    file_embed = 'word_embedding'
    get_word_embedding(embedding_size, file_embed)

