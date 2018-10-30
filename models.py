#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:36:52 2018

@author: heiko
"""

# Models for MNIST, basic stuff

import tensorflow as tf
import numpy as np






# models for imagenet -> find the names of the fitting tensors

with tf.Session(graph=tf.Graph()) as sess: 
    pass
sess = tf.Session()
rMtest = tf.saved_model.loader.load(sess, ['serve'], 
    '/Users/heiko/Google Drive/models/official/resnet/resnet_v2_fp32_savedmodel_NCHW/1538687196')
