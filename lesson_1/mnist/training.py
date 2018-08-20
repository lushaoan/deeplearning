# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2018.08.18'
__copyright__ = 'Copyright 2018, LSA'
__all__ = []


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import model


batch_size = 128


if __name__ == '__main__':
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1], name='input-x')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input-y')

    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y = model.inference(inputTensor=x, train=True, regularizer=regularizer)
    loss = model.losses(logits=y, labels=y_)
    train_op = model.training(loss=loss)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./log",sess.graph)
        saver = tf.train.Saver()
        for step in range(0, 5000):
            xs, ys = mnist.train.next_batch(batch_size)
            reshaped_xs = np.reshape(xs, [batch_size, 28, 28, 1])
            _, summary_str, loss_value = sess.run([train_op, summary_op, loss], feed_dict={x:reshaped_xs, y_:ys})

            if step%50 == 0:
                writer.add_summary(summary_str, step)

            if step%100 == 0:
                print(loss_value)

            if step%800 == 0:
                saver.save(sess=sess, save_path='./save_data/model', global_step=step)

        writer.close()