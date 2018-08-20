# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2018.08.20'
__copyright__ = 'Copyright 2018, LSA'
__all__ = []


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('save_data/model-4800.meta')
        saver.restore(sess, tf.train.latest_checkpoint('save_data/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input-x:0')

        xs, ys = mnist.train.next_batch(128)
        reshaped_xs = np.reshape(xs, [128, 28, 28, 1])
        feed_dict = {x:reshaped_xs}

        op_to_restore = graph.get_tensor_by_name('layer7-softmax/softmax:0')

        print(sess.run(op_to_restore, feed_dict))