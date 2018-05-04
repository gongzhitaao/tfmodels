import logging

import numpy as np
import tensorflow as tf


__all__ = ['load_data', 'build_metric']


logger = logging.getLogger(__name__)
info = logger.info


class DisableLogger():
    def __enter__(self):
        logging.disable(100000)

    def __exit__(self, *_):
        logging.disable(logging.NOTSET)


def load_data(data, bipolar, validation_split=0.1):
    d = np.load(data)
    ret = []

    def _load(d, name):
        X_data, y_data = d['X_{}'.format(name)], d['y_{}'.format(name)]
        y_data = np.expand_dims(y_data, axis=1)
        if bipolar:
            y_data = 2 * y_data - 1
        return (X_data, y_data)

    if 'X_train' in d:
        X_train, y_train = _load(d, 'train')
        ret.append((X_train, y_train))
        info('X_train shape: {}'.format(X_train.shape))
        info('y_train shape: {}'.format(y_train.shape))

    if 'X_test' in d:
        X_test, y_test = _load(d, 'test')
        ret.append((X_test, y_test))
        info('X_test shape: {}'.format(X_test.shape))
        info('y_test shape: {}'.format(y_test.shape))

    if 'X_train' in locals() and validation_split > 0:
        ind = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[ind], y_train[ind]
        n = int(X_train.shape[0] * validation_split)
        X_valid = X_train[:n]
        X_train = X_train[n:]
        y_valid = y_train[:n]
        y_train = y_train[n:]
        ret[0] = (X_train, y_train)
        ret.append((X_valid, y_valid))

    if len(ret) > 1:
        return tuple(ret)
    return ret[0]


def build_metric(env, cfg):
    if cfg.output == tf.sigmoid:
        y = tf.to_float(env.y)
        with tf.variable_scope('acc'):
            t0 = tf.greater(env.ybar, 0.5)
            t1 = tf.greater(y, 0.5)
            count = tf.equal(t0, t1)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y, logits=env.model.logits)
            env.loss = tf.reduce_mean(xent)
    elif cfg.output == tf.tanh:
        y = tf.to_float(env.y)
        with tf.variable_scope('acc'):
            t0 = tf.greater(env.ybar, 0.0)
            t1 = tf.greater(y, 0.0)
            count = tf.equal(t0, t1)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            env.loss = tf.losses.mean_squared_error(
                labels=y, predictions=env.ybar,
                reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif cfg.output == tf.nn.softmax:
        y = tf.one_hot(env.y, cfg.n_classes, on_value=1.0, off_value=0.0)
        with tf.variable_scope('acc'):
            ybar = tf.argmax(env.ybar, axis=1, output_type=tf.int32)
            count = tf.equal(tf.reshape(env.y, [-1]), ybar)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y, logits=env.model.logits)
            env.loss = tf.reduce_mean(xent)
    else:
        raise ValueError('Unknown output function')
    return env
