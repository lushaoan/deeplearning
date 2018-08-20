# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2018.08.15'
__copyright__ = 'Copyright 2018, LSA'
__all__ = []


import tensorflow as tf


def inference(inputTensor, regularizer, train):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(name='weight', shape=[5,5,1,32], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable(name='bias', shape=[32], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        conv1 = tf.nn.conv2d(input=inputTensor, filter=conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(features=tf.nn.bias_add(value=conv1, bias=conv1_bias))

        # tf.summary.histogram(name='/weight_hist', values=conv1_weights)
        tf.summary.image(name='/relu1', tensor=relu1[0:1,:,:,0:1])

    with tf.variable_scope('layer2-poo1'):
        pool1 = tf.nn.max_pool(value=relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable(name='weight', shape=[5,5,32,64], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable(name='bias', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weight, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(features=tf.nn.bias_add(value=conv2, bias=conv2_bias))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(value=relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer5-fc1'):
        pool2_shape = pool2.get_shape().as_list()
        dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        reshaped = tf.reshape(tensor=pool2, shape=[pool2_shape[0], -1])

        fc1_weight = tf.get_variable(name='weight', shape=[dim, 512], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
        fc1_bias = tf.get_variable(name='bias', shape=[512], dtype=tf.float32,
                                   initializer=tf.constant_initializer(value=0.0))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weight))

        fc1 = tf.nn.relu(features=tf.matmul(a=reshaped, b=fc1_weight)+fc1_bias)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable(name='weight', shape=[512, 128], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weight))

        fc2_bias = tf.get_variable(name='bias', shape=[128])

        fc2 = tf.nn.relu(features=tf.matmul(a=fc1, b=fc2_weight)+fc2_bias)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer7-softmax'):
        weight = tf.get_variable(name='weight', shape=[128, 10], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias = tf.get_variable(name='bias', shape=[10], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.1))
        softmax = tf.add(x=tf.matmul(a=fc2, b=weight), y=bias, name='softmax')

    return softmax


def losses(logits, labels):
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels, 1), logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) #'loss/add:0'
        print(loss)

        tf.summary.histogram(name='/loss', values=loss)

    return loss


def training(loss):
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        variable_average = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
        variable_average_op = variable_average.apply(tf.trainable_variables())

        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss=loss, global_step=global_step)
        train_op = tf.group(train_step, variable_average_op)

    return train_op