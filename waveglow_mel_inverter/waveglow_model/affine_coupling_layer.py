#!/usr/bin/env python
# coding: utf-8

# # Invertible Convolution and WaveNet Custom Layers 

import tensorflow as tf
from tensorflow.keras import layers

import os, sys


# ## Custom Affine Coupling Layer
# 
# This layer does not have any trainable weights. It can be inverted by setting the training boolean to false.
class AffineCoupling(layers.Layer):
    """
    Invertible Affine Layer
    
    The inverted behaviour is obtained by setting the training boolean
    in the call method to false
    """

    def __init__(self, **kwargs):
        super(AffineCoupling, self).__init__(**kwargs)

    def call(self, inputs, training=None):

        audio_1, wavenet_output = inputs

        log_s, bias = tf.split(wavenet_output, 2, axis=-1)

        if training:
            audio_1 = audio_1 * tf.math.exp(log_s) + bias
            loss = - tf.reduce_sum(log_s)
            self.add_loss(loss)
            tf.summary.scalar(name='loss', data=loss)
        else:
            audio_1 = (audio_1 - bias) * tf.math.exp(- log_s)

        return audio_1

    def get_config(self):
        config = super(AffineCoupling, self).get_config()
        return config


