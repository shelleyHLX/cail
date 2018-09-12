# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import shutil
import time
from utils import get_logger
import network

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('../..')
from data_helper import to_categorical
from evaluator import cail_evaluator

flags = tf.flags
flags.DEFINE_bool('is_retrain', False, 'if is_retrain is true, not rebuild the summary')
flags.DEFINE_integer('max_epoch', 1, 'update the embedding after max_epoch, default: 1')
flags.DEFINE_integer('max_max_epoch', 1000, 'all training epoches, default: 1000')
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_float('decay_rate', 0.6, 'decay rate, default: 0.65')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob for training, default: 0.5')
flags.DEFINE_string("log_file_train",     "train.log",    "File for log")
flags.DEFINE_integer('decay_step', 5000, 'decay_step, default: 5000')
flags.DEFINE_integer('valid_step', 2500, 'valid_step, default: 2500')
flags.DEFINE_float('last_score12', 0.0, 'if valid_score12 > last_score12, save new model. default: 0.0')

FLAGS = flags.FLAGS

lr = FLAGS.lr
last_score12 = FLAGS.last_score12
settings = network.Settings()
summary_path = settings.summary_path
ckpt_path = settings.ckpt_path
model_path = ckpt_path + 'model.ckpt'
log_path = settings.log_path

embedding_path = '../../data/word_embedding_256.npy'

data_train_path = '../../data/wd_pdQS200/train/'
data_valid_path = '../../data/wd_pdQS200/valid/'

tr_batches = os.listdir(data_train_path)
va_batches = os.listdir(data_valid_path)

n_tr_batches = len(tr_batches)
n_va_batches = len(va_batches)



def get_batch(data_path, batch_id):
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    return [X_batch, y_batch]

def valid_epoch(data_path, sess, model):
    va_batches = os.listdir(data_path)
    n_va_batches = len(va_batches)
    _costs = 0.0
    predict_labels_list = list()
    marked_labels_list = list()
    for i in range(n_va_batches):
        [X_batch, y_batch] = get_batch(data_path, i)
        marked_labels_list.extend(y_batch)
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        fetches = [model.loss, model.y_pred]
        feed_dict = {model.X_inputs: X_batch,
                     model.y_inputs: y_batch, model.batch_size: _batch_size,
                     model.tst: True, model.keep_prob: 1.0}
        _cost, predict_labels = sess.run(fetches, feed_dict)
        _costs += _cost
        predict_labels_list.extend(predict_labels)
    f1_micro, f1_macro, score12 = cail_evaluator(predict_labels_list, marked_labels_list)
    return f1_micro, f1_macro, score12

def train_epoch(data_path, sess, model, train_fetches,
                valid_fetches, train_writer, test_writer, logger):
    global last_score12
    global lr
    time0 = time.time()
    batch_indexs = np.random.permutation(n_tr_batches)
    for batch in tqdm(range(n_tr_batches)):
        global_step = sess.run(model.global_step)
        if 0 == (global_step + 1) % FLAGS.valid_step:
            f1_micro, f1_macro, score12 = valid_epoch(data_valid_path, sess, model)
            print('Global_step=%d: f1_micro=%g, f1_macro=%g, score12=%g, time=%g s' % (
                global_step, f1_micro, f1_macro, score12, time.time() - time0))
            logger.info('END:Global_step={}: f1_micro={}, f1_macro={}, score12={}'.
                        format(sess.run(model.global_step), f1_micro, f1_macro, score12))
            time0 = time.time()
            if score12 > last_score12:
                last_score12 = score12
                saving_path = model.saver.save(sess, model_path, global_step+1)
                print('saved new model to %s ' % saving_path)
        # training
        batch_id = batch_indexs[batch]
        [X_batch, y_batch] = get_batch(data_path, batch_id)
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        feed_dict = {model.X_inputs: X_batch,
                     model.y_inputs: y_batch, model.batch_size: _batch_size,
                     model.tst: False, model.keep_prob: FLAGS.keep_prob}
        summary, _cost, _, _ = sess.run(train_fetches, feed_dict)  # the cost is the mean cost of one batch
        # valid per 500 steps
        if 0 == (global_step + 1) % 500:
            train_writer.add_summary(summary, global_step)
            batch_id = np.random.randint(0, n_va_batches)  # 随机选一个验证batch
            [X_batch, y_batch] = get_batch(data_valid_path, batch_id)
            y_batch = to_categorical(y_batch)
            _batch_size = len(y_batch)
            feed_dict = {model.X_inputs: X_batch,
                         model.y_inputs: y_batch, model.batch_size: _batch_size,
                         model.tst: True, model.keep_prob: 1.0}
            summary, _cost = sess.run(valid_fetches, feed_dict)
            test_writer.add_summary(summary, global_step)


def main(_):
    global ckpt_path
    global last_score12
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    elif not FLAGS.is_retrain:
        shutil.rmtree(summary_path)
        os.makedirs(summary_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print('1.Loading data...')
    W_embedding = np.load(embedding_path)
    print('training sample_num = %d' % n_tr_batches)
    print('valid sample_num = %d' % n_va_batches)
    logger = get_logger(log_path + FLAGS.log_file_train)

    print('2.Building model...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = network.Atten_TextCNN(W_embedding, settings)
        with tf.variable_scope('training_ops') as vs:
            learning_rate = tf.train.exponential_decay(FLAGS.lr, model.global_step,
                                                       FLAGS.decay_step,
                                                       FLAGS.decay_rate, staircase=True)
            with tf.variable_scope('Optimizer1'):
                tvars1 = tf.trainable_variables()
                grads1 = tf.gradients(model.loss, tvars1)
                optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op1 = optimizer1.apply_gradients(zip(grads1, tvars1),
                                                   global_step=model.global_step)
            with tf.variable_scope('Optimizer2'):
                tvars2 = [tvar for tvar in tvars1 if 'embedding' not in tvar.name]
                grads2 = tf.gradients(model.loss, tvars2)
                optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op2 = optimizer2.apply_gradients(zip(grads2, tvars2),
                                                   global_step=model.global_step)
            update_op = tf.group(*model.update_emas)
            merged = tf.summary.merge_all()  # summary
            train_writer = tf.summary.FileWriter(summary_path + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(summary_path + 'test')
            training_ops = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

        if os.path.exists(ckpt_path + "checkpoint"):
            print("Restoring Variables from Checkpoint...")
            model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            f1_micro, f1_macro, score12 = valid_epoch(data_valid_path, sess, model)
            print('f1_micro=%g, f1_macro=%g, score12=%g' % (f1_micro, f1_macro, score12))
            sess.run(tf.variables_initializer(training_ops))
            train_op2 = train_op1
        else:
            print('Initializing Variables...')
            sess.run(tf.global_variables_initializer())

        print('3.Begin training...')
        print('max_epoch=%d, max_max_epoch=%d' % (FLAGS.max_epoch, FLAGS.max_max_epoch))
        logger.info('max_epoch={}, max_max_epoch={}'.format(FLAGS.max_epoch, FLAGS.max_max_epoch))
        train_op = train_op2
        for epoch in range(FLAGS.max_max_epoch):
            print('\nepoch: ', epoch)
            logger.info('epoch:{}'.format(epoch))
            global_step = sess.run(model.global_step)
            print('Global step %d, lr=%g' % (global_step, sess.run(learning_rate)))
            if epoch == FLAGS.max_epoch:
                train_op = train_op1

            train_fetches = [merged, model.loss, train_op, update_op]
            valid_fetches = [merged, model.loss]
            train_epoch(data_train_path, sess, model, train_fetches,
                        valid_fetches, train_writer, test_writer, logger)
        # 最后再做一次验证
        f1_micro, f1_macro, score12 = valid_epoch(data_valid_path, sess, model)
        print('END:Global_step=%d: f1_micro=%g, f1_macro=%g, score12=%g' % (
            sess.run(model.global_step), f1_micro, f1_macro, score12))
        logger.info('END:Global_step={}: f1_micro={}, f1_macro={}, score12={}'.
                    format(sess.run(model.global_step), f1_micro, f1_macro, score12))
        if score12 > last_score12:
            saving_path = model.saver.save(sess, model_path, sess.run(model.global_step)+1)
            print('saved new model to %s ' % saving_path)
            logger.info('saved new model to {}'.format(saving_path))


if __name__ == '__main__':
    tf.app.run()
