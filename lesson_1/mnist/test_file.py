# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2018.08.13'
__copyright__ = 'Copyright 2018, LSA'
__all__ = []


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2


# def get_batch()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    x, y = mnist.train.next_batch(5)
    img = tf.reshape(x, [5,28,28,1])
    print(type(img[0]))
    print(img.shape)
    print(img[0])
    cv2.imshow('img', img[0])
    cv2.waitKey(0)