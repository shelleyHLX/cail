# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import time
import network
from utils import get_logger

sys.path.append('../..')
from evaluator import cail_evaluator

settings = network.Settings()
model_name = settings.model_name
ckpt_path = settings.ckpt_path

scores_path = '../../scores/'

if not os.path.exists(scores_path):
    os.makedirs(scores_path)

embedding_path = '../../data/word_embedding_256.npy'

data_valid_path = '../../data/wd_pdQS200/valid/'
data_test_path = '../../data/wd_pdQS200/test/'

va_batches = os.listdir(data_valid_path)
te_batches = os.listdir(data_test_path)
n_va_batches = len(va_batches)
n_te_batches = len(te_batches)

def get_batch(data_path, batch_id):
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    return [X_batch, y_batch]

def predict(sess, model, logger):
    time0 = time.time()
    te_batches = os.listdir(data_test_path)
    n_te_batches = len(te_batches)
    predict_labels_list = list()  # 所有的预测结果
    marked_labels_list = list()
    for i in tqdm(range(n_te_batches)):
        X_batch, y_batch = get_batch(data_test_path, i)
        _batch_size = len(X_batch)
        marked_labels_list.extend(y_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs: X_batch,
                     model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_labels_list.extend(predict_labels)

    predict_scores_file = scores_path + model_name + '/' + 'predict.npy'
    marked_scores_file = scores_path + model_name + '/' + 'origin.npy'
    np.save(predict_scores_file, predict_labels_list)
    np.save(marked_scores_file, marked_labels_list)
    print('save predict_labels_list', predict_scores_file)

    f1_micro, f1_macro, score12 = cail_evaluator(predict_labels_list, marked_labels_list)
    print('f1_micro=%g, f1_macro=%g, score12=%g, time=%g s'
          % (f1_micro, f1_macro, score12, time.time() - time0))
    logger.info('\nTest predicting...\nEND:Global_step={}: f1_micro={}, f1_macro={}, score12={}, time={} s'.
                format(sess.run(model.global_step), f1_micro, f1_macro, score12, time.time() - time0))

def predict_valid(sess, model, logger):
    """Test on the valid data."""
    time0 = time.time()
    predict_labels_list = list()
    marked_labels_list = list()
    for i in tqdm(range(int(n_va_batches))):
        [X_batch, y_batch] = get_batch(data_valid_path, i)
        marked_labels_list.extend(y_batch)
        _batch_size = len(X_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs: X_batch,
                     model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_labels_list.extend(predict_labels)

    f1_micro, f1_macro, score12 = cail_evaluator(predict_labels_list, marked_labels_list)
    print('precision_micro=%g, recall_micro=%g, score12=%g, time=%g s'
          % (f1_micro, f1_macro, score12, time.time() - time0))
    logger.info('\nValid predicting...\nEND:Global_step={}: f1_micro={}, f1_macro={}, score12={}, time={} s'.
                format(sess.run(model.global_step), f1_micro, f1_macro, score12, time.time() - time0))

def main(_):
    if not os.path.exists(ckpt_path + 'checkpoint'):
        print('there is not saved model, please check the ckpt path')
        exit()
    print('Loading model...')
    W_embedding = np.load(embedding_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log_path = scores_path + settings.model_name + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = get_logger(log_path + 'predict.log')
    with tf.Session(config=config) as sess:
        model = network.Atten_TextCNN(W_embedding, settings)
        model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        print('Valid predicting...')
        predict_valid(sess, model, logger)

        print('Test predicting...')
        predict(sess, model, logger)


if __name__ == '__main__':
    tf.app.run()

