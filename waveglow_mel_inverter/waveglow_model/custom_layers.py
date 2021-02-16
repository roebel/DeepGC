#!/usr/bin/env python
# coding: utf-8

# # Invertible Convolution and WaveNet Custom Layers 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
try:
    from .affine_coupling_layer import AffineCoupling
except (ImportError, ModuleNotFoundError):
    from affine_coupling_layer import AffineCoupling

import os, sys


# ## Invertible Convolution
# 
# The training boolean in the call method can be used to run the layer in reverse. 
#  This layer should be wrapped in tensorflow-addons weight_norm layer in the waveglow initialisation call.

class Inv1x1Conv(layers.Conv1D):
    """
    Tensorflow 2.0 implementation of the inv1x1conv layer 
    directly subclassing the tensorflow Conv1D layer
    """

    def __init__(self, filters, **kwargs):
        super(Inv1x1Conv, self).__init__(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.initializers.orthogonal(),
            activation="linear",
            **kwargs)
        self._built_input_shape = None
        self.kernel_needs_update = tf.constant(True, tf.bool)

    @property
    def output_shape(self):
        if self._built_input_shape is not None:
            return super().compute_output_shape(self._built_input_shape)

    def build(self, input_shape):
        self._built_input_shape = input_shape
        super().build(input_shape)
        self.kernel_needs_update = tf.constant(True, tf.bool)

    def update_inverse(self):
        """
        sync backward kernel
        """
        self.kernel_inverse = tf.cast(tf.linalg.inv(
            tf.cast(self.kernel, tf.float64)), dtype=self.dtype)
        self.kernel_needs_update = tf.constant(False, tf.bool)

    def update_forward(self):
        self.kernel = tf.cast(tf.linalg.inv(
                tf.cast(self.kernel_inverse, tf.float64)), dtype=self.dtype)

    def call(self, inputs, training=True):
        if training:
            sign, log_det_weights = tf.linalg.slogdet( tf.cast(self.kernel, tf.float32))
            # this describes expansion/contruction in groups dimension
            # for all other dimensions this will be the same so we can simply multiply to get the overall 
            # expansion or contraction
            loss = - inputs.shape[0] * inputs.shape[1] * tf.cast(tf.reduce_sum(log_det_weights), dtype=self.dtype)
            self.add_loss(loss)
            tf.summary.scalar(name='loss', data=loss)
            self.kernel_needs_update = tf.constant(True, tf.bool)
            return super(Inv1x1Conv, self).call(inputs)
        else:
            if self.kernel_needs_update:
                self.update_inverse()
            return tf.nn.conv1d(inputs, self.kernel_inverse,
                                stride=1, padding='SAME')


# ## Invertible Convolution
#
# The training boolean in the call method can be used to run the layer in reverse.
#  This layer should be wrapped in tensorflow-addons weight_norm layer in the waveglow initialisation call.

class Inv1x1Conv_SVD(layers.Conv1D):
    """
    Tensorflow 2.0 implementation of the inv1x1conv layer
    directly subclassing the tensorflow Conv1D layer
    """

    def __init__(self, filters, **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.initializers.orthogonal(),
            activation="linear",
            **kwargs)
        self._built_input_shape = None

    @property
    def output_shape(self):
        if self._built_input_shape is not None:
            return super().compute_output_shape(self._built_input_shape)

    def build(self, input_shape):
        self._built_input_shape = input_shape
        super().build(input_shape)

        s, u, v = tf.linalg.svd(self.kernel)
        kernel_shape = self.kernel.shape
        self.kernel = None

        # note the internal name is mis leading, tf.linalg.svd does not return the transposition of v!
        self.u_v = self.add_weight(name="u_vT",
                                 shape=[2*kernel_shape[0], kernel_shape[1], kernel_shape[2]],
                                 initializer=tf.keras.initializers.get('ones'),
                                 dtype=self.dtype,
                                 trainable=True)

        self.s = self.add_weight( name="s",
                                  shape=[kernel_shape[0], 1, kernel_shape[2]],
                                  initializer=tf.keras.initializers.get('ones'),
                                  dtype=self.dtype,
                                  trainable=True)

        self.s.assign(tf.expand_dims(s, 1))
        self.u_v.assign(tf.concat((u, v), axis=0))

    def call(self, inputs, training=True):
        q,r = tf.linalg.qr(self.u_v)
        # resolve ambiguity in qr decomposition
        # see reorthogonalization schemes in https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf
        # and answer with comment in https://math.stackexchange.com/questions/1960072/uniqueness-of-qr-decomposition
        # or more comprhensive https://math.stackexchange.com/questions/989627/uniqueness-of-thin-qr-factorization
        q = tf.expand_dims(tf.sign(tf.linalg.diag_part(r)), 1) * q
        if training:
            # this describes expansion/construction in groups dimension
            # for all other dimensions this will be the same so we can simply multiply to get the overall
            # expansion or contraction
            loss = - inputs.shape[0] * inputs.shape[1] * tf.reduce_sum(tf.math.log(self.s))
            self.add_loss(loss)
            tf.summary.scalar(name='loss', data=loss)
            self.kernel = tf.matmul(q[:self.s.shape[0]],
                                    self.s * q[self.s.shape[0]:], transpose_b=True)
        else:
            self.kernel = tf.matmul(q[self.s.shape[0]:],
                                    q[:self.s.shape[0]] / self.s, transpose_b=True)
        return super().call(inputs)


class Inv1x1MM_PCA(layers.Layer):
    """
    Tensorflow 2.0 implementation of the inv1x1conv layer
    directly subclassing the tensorflow Conv1D layer
    """

    def __init__(self, **kwargs):
        if "trainable" not in kwargs:
            super().__init__(trainable=False, **kwargs)
        else:
            super().__init__( **kwargs)

        self._built_input_shape = None
        if self.trainable:
            raise RuntimeError("Inv1x1MM_PCA::warning::training the whitening layer not yet supported, "
                               "weight updates will not preserve orthogonality of the projection matrix!")
    @property
    def output_shape(self):
        return self._built_input_shape

    def build(self, input_shape):
        self._built_input_shape = input_shape
        # note the internal name is mis leading, tf.linalg.svd does not return the transposition of v!
        self.orth_mat = self.add_weight(name="orth_mat",
                                 shape=[1, input_shape[-1], input_shape[-1]],
                                 initializer=tf.keras.initializers.orthogonal(),
                                 dtype=self.dtype,
                                 trainable=self.trainable)

        self.scale = self.add_weight( name="scale",
                                  shape=[1, 1, input_shape[-1]],
                                  initializer=tf.keras.initializers.Constant(0.5),
                                  dtype=self.dtype,
                                  trainable=self.trainable)
        self.mean = self.add_weight( name="mean",
                                  shape=[1, 1, input_shape[-1]],
                                  initializer=tf.keras.initializers.Constant(0.),
                                  dtype=self.dtype,
                                  trainable=self.trainable)

        super().build(input_shape)

    def set_weights_from_cov(self, cov, mean):
        """
        produce waights parameters that produce whiening effect on data with gien covariance matrix
        """
        if not self.built:
            raise RuntimeError("Inv1x1MM_PCA::error::you cannot set weights for a network that has not yet been built, "
                               "the weights you set will be overwritten!")
        eigvals, eigvecs = np.linalg.eig(cov)
        self.scale.assign( np.expand_dims(np.sqrt(eigvals), 1))
        self.orth_mat.assign(eigvecs)
        self.mean.assign(mean)

    def call(self, inputs, training=True):
        # debug = False
        if training:
            # this describes expansion/construction in groups dimension
            # for all other dimensions this will be the same so we can simply multiply to get the overall
            # expansion or contraction
            loss = - inputs.shape[0] * inputs.shape[1] * tf.reduce_sum(- tf.math.log(self.scale))
            self.add_loss(loss)
            tf.summary.scalar(name='loss', data=loss)
            kernel = self.orth_mat/ self.scale
            # if debug:
            #     ikernel = tf.transpose(self.orth_mat * self.scale, (0, 2, 1))
            #     print(f"MMPCA: train on kern inv error {tf.reduce_mean(tf.abs(tf.matmul(kernel, ikernel) - tf.eye(8)))}")

            res = tf.matmul(inputs - self.mean, kernel)
            # if debug:
            #     dd = tf.matmul(res, ikernel) + self.mean
            #     print(f"MMPCA: dinv err = {tf.reduce_mean(tf.abs(inputs - dd))}")
        else:
            kernel = tf.transpose(self.orth_mat * self.scale, (0, 2, 1))
            # if debug:
            #     ikernel = self.orth_mat / self.scale
            #     print(f"MMPCA: train off kern inv error {tf.reduce_mean(tf.abs(tf.matmul(kernel, ikernel) - tf.eye(8)))}")

            res = tf.matmul(inputs, kernel) + self.mean
            # if debug:
            #     dd = tf.matmul(res - self.mean, ikernel)
            #     print(f"PCA: dinv err = {tf.reduce_mean(tf.abs(inputs - dd))}")
        return res


class Inv1x1MM_SVD(layers.Layer):
    """
    Tensorflow 2.0 implementation of the inv1x1conv layer
    directly subclassing the tensorflow Conv1D layer
    """

    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self._built_input_shape = None

    @property
    def output_shape(self):
        return self._built_input_shape

    def build(self, input_shape):
        self._built_input_shape = input_shape[0]
        super().build(input_shape[0])

    def call(self, inputs, training=True):
        # split signal from kernel
        data, paras = inputs
        ls = tf.reshape(paras[:,:,:data.shape[-1]], (paras.shape[0], paras.shape[1], 1, data.shape[-1]))
        s = tf.exp(ls)
        u_v =  tf.reshape(paras[:,:,data.shape[-1]:], (paras.shape[0], paras.shape[1], 2, data.shape[-1], data.shape[-1]))
        q,r = tf.linalg.qr(u_v)

        # resolve ambiguity in qr decomposition
        # see reorthogonalization schemes in https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf
        # and answer with comment in https://math.stackexchange.com/questions/1960072/uniqueness-of-qr-decomposition
        # or more comprhensive https://math.stackexchange.com/questions/989627/uniqueness-of-thin-qr-factorization

        #print(f"q.shape {q.shape} r.shape {r.shape} diag shape {tf.linalg.diag_part(r).shape} expanded r {tf.expand_dims(tf.sign(tf.linalg.diag_part(r)), 3).shape}")
        q = tf.expand_dims(tf.sign(tf.linalg.diag_part(r)), 3) * q
        #print(f"mean q {tf.reduce_mean(q)} mean abs q {tf.reduce_mean(tf.abs(q))} mean s {tf.reduce_mean(s)}")
        #if False:
        #    print(f"final q shape {q.shape} s.shape {s.shape} min s {tf.reduce_min(s)}")
        #    print(f"qr rec err = {tf.reduce_mean(tf.abs(tf.matmul(q, tf.expand_dims(tf.sign(tf.linalg.diag_part(r)), 4) * r) - u_v))}")
        if training:
            # this describes expansion/construction in groups dimension
            # for all other dimensions this will be the same so we can simply multiply to get the overall
            # expansion or contraction
            loss = - tf.reduce_sum(ls)
            self.add_loss(loss)
            tf.summary.scalar(name='loss', data=loss)
            kernel = tf.matmul(q[:, :, 0], s * q[:, :, 1], transpose_b=True)
            #ikernel = tf.matmul(q[:, :, 1], q[:, :, 0] / s, transpose_b=True)
            #if False:
            #    print(f"orthog err u {tf.reduce_mean(tf.abs(tf.matmul(q[:, :, 0], q[:, :, 0], transpose_b=True) - tf.eye(8)))}", end=", ")
            #    print(f"v {tf.reduce_mean(tf.abs(tf.matmul(q[:, :, 1], q[:, :, 1], transpose_b=True) - tf.eye(8)))}")
            #    print(f"kern inv error {tf.reduce_mean(tf.abs(tf.matmul(kernel, ikernel) - tf.eye(8)))}")
        else:
            kernel = tf.matmul(q[:, :, 1], q[:, :, 0] / s, transpose_b=True)
            #ikernel = tf.matmul(q[:, :, 0], q[:, :, 1] * s, transpose_b=True)
            #if False:
            #    print(f"orthog err u:{tf.reduce_mean(tf.abs(tf.matmul(q[:, :, 0], q[:, :, 0], transpose_b=True) - tf.eye(8)))}", end =", ")
            #    print(f"v: {tf.reduce_mean(tf.abs(tf.matmul(q[:, :, 1], q[:, :, 1], transpose_b=True) - tf.eye(8)))}")
            #    print(f"kern inv error {tf.reduce_mean(tf.abs(tf.matmul(kernel, ikernel) - tf.eye(8)))}")
        res = tf.squeeze(tf.matmul(tf.expand_dims(data, 2), kernel), axis=2)
        #dd = tf.squeeze(tf.matmul(tf.expand_dims(res, 2), ikernel), axis=2)
        #print(f"data.shape {data.shape} kernel.shape {kernel.shape},"
        #      f" abs mean data: {tf.reduce_mean(tf.abs(data))} paras: {tf.reduce_mean(tf.abs(paras))} res: {tf.reduce_mean(tf.abs(res))}")
        #print(f"dinv err = {tf.reduce_mean(tf.abs(data - dd))}")
        return res



class SpectralUpsampling(layers.Layer):
    def __init__(self, filters, kernel_sizes, activation_functions, name="spect_upsampling", **kwargs):

        super().__init__(**kwargs)
        self.layers = []
        self.filters = filters
        self.activation_functions = activation_functions
        self.kernel_sizes = kernel_sizes
        self._output_shape = None
        for nc, act, ks in zip(self.filters, self.activation_functions, self.kernel_sizes):
            self.layers.append(layers.Conv1D(nc, ks, padding="SAME", dtype=self.dtype, name=name))
            if act is not None:
                if act.startswith("p"):
                    alpha_init=0.5
                    if "," in act:
                        alpha_init=float(act.split(",")[1])
                    self.layers.append(tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(alpha_init),  shared_axes=1))
                elif act.startswith("l"):
                    alpha_init=0.5
                    if "," in act:
                        alpha_init=float(act.split(",")[1])
                    self.layers.append(tf.keras.layers.LeakyReLU(alpha=alpha_init))
                elif act.startswith("r"):
                    self.layers.append(tf.keras.layers.ReLU())

    @property
    def output_shape(self):
        return self._output_shape

    def build(self, input_shape):
        for ll in self.layers:
            ll.build(input_shape=input_shape)
            input_shape = ll.compute_output_shape(input_shape=input_shape)
        self._output_shape = input_shape

    def call(self, inputs, **kwargs):
        res = inputs
        for ll in self.layers:
            res = ll.call(res)
        return res

    def get_config(self):
        config = super().get_config()
        config.update(num_channels=self.num_channels)
        config.update(activation_functions=self.activation_functions)
        config.update(kernel_sizes=self.kernel_sizes)
        return config


# ## Custom Implementation of Conv1D with weight normalization
class Conv1DWeightNorm(layers.Conv1D):

    def __init__(self, filters, use_weight_norm=True,   **kwargs):
        super().__init__(
            filters = filters,
            activation="linear",
            **kwargs)
        self.filters = filters
        self.use_weight_norm = use_weight_norm
        self._built_input_shape = None

    @property
    def output_shape(self):
        if self._built_input_shape is not None:
            return super().compute_output_shape(self._built_input_shape)

    def build(self, input_shape):
        self._built_input_shape = input_shape
        super().build(input_shape)
        if self.use_weight_norm:
            self.layer_depth = self.filters
            self.kernel_norm_axes = [0, 1]

            self.v = self.kernel
            self.kernel = None
            self.g = self.add_weight(
                name="g",
                shape=self.layer_depth,
                initializer=tf.keras.initializers.get('ones'),
                dtype=self.dtype,
                trainable=True)

            self.g.assign(tf.linalg.norm(self.v, axis=self.kernel_norm_axes))

    def call(self, inputs, **kwargs):
        if self.use_weight_norm:
            self.kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g
        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(use_weight_norm=self.use_weight_norm)
        return config


# ## Nvidia WaveNet Implementation
# Difference with the original implementations :
# WaveNet convonlution need not be causal. 
# No dilation size reset. 
# Dilation doubles on each layer
# 
# It could be worth investigating whether including the weight_norm wrapper of tensorflow addon incurs significant improvements during training
class WaveNetNvidia(layers.Layer):
    """
    Wavenet Block as defined in the WaveGlow implementation from Nvidia
    
    WaveNet convonlution need not be causal. 
    No dilation size reset. 
    Dilation doubles on each layer.
    """

    def __init__(self, n_in_channels, n_channels=256,
                 n_layers=12, kernel_size=3, n_out_channels = None,
                 dilation_rate_step = 1, max_log2_dilation_rate = None, use_weight_norm=True, **kwargs):
        super(WaveNetNvidia, self).__init__(**kwargs)

        assert (kernel_size % 2 == 1)
        assert (n_channels % 2 == 0)

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_in_channels = n_in_channels
        self.kernel_size = kernel_size
        self.dilation_rate_step = dilation_rate_step
        self.max_log2_dilation_rate = max_log2_dilation_rate
        self.in_layers = []
        self.normalisation_layers = []
        self.res_skip_layers = []
        self.cond_layers = []
        self.n_out_channels = 2 * self.n_in_channels
        if n_out_channels is not None:
          self.n_out_channels = n_out_channels
        self._built_input_shape = None
        self.use_weight_norm = use_weight_norm

        self.start = Conv1DWeightNorm(filters=self.n_channels,
                                      kernel_size=1,
                                      dtype=self.dtype,
                                      use_weight_norm=self.use_weight_norm,
                                      name="start")

        self.end = layers.Conv1D( filters=self.n_out_channels,
                                     kernel_size=1,
                                     kernel_initializer=tf.initializers.zeros(),
                                     bias_initializer=tf.initializers.zeros(),
                                     dtype=self.dtype,
                                     name="end")

        self.cond_layer = Conv1DWeightNorm(filters=2 * self.n_channels * self.n_layers,
                                           kernel_size=1,
                                           dtype=self.dtype,
                                           use_weight_norm=self.use_weight_norm,
                                           name="cond_")

        for index in range(self.n_layers):
            if max_log2_dilation_rate is not None:
                dilation_rate = 2 ** (int(index//self.dilation_rate_step) % self.max_log2_dilation_rate)
            else:
                dilation_rate = 2 ** int(index//self.dilation_rate_step)

            in_layer = Conv1DWeightNorm(filters=2 * self.n_channels,
                                        kernel_size=self.kernel_size,
                                        dilation_rate=dilation_rate,
                                        padding="SAME",
                                        dtype=self.dtype,
                                        use_weight_norm=self.use_weight_norm,
                                        name="conv1D_{}".format(index))

            # Nvidia has a weight_norm func here, training stability?
            # Memory expensive in implementation of tf-addons wrapper
            self.in_layers.append(in_layer)
            if index < self.n_layers - 1:
                res_skip_channels = 2 * self.n_channels
            else:
                res_skip_channels = self.n_channels

            res_skip_layer = Conv1DWeightNorm(filters=res_skip_channels,
                                              kernel_size=1,
                                              dtype=self.dtype,
                                              use_weight_norm=self.use_weight_norm,
                                              name="res_skip_{}".format(index))

            self.res_skip_layers.append(res_skip_layer)

    def call(self, inputs, **_):
        """
        This implementatation does not require exposing a training boolean flag 
        as only the affine coupling behaviour needs reversing during
        inference.
        """
        audio_0, spect = inputs

        started = self.start(audio_0)
        cond_layers = tf.split(self.cond_layer(spect), self.n_layers, axis=-1)

        for index in range(self.n_layers):
            in_layered = self.in_layers[index](started)
            half_tanh, half_sigmoid = tf.split(in_layered  + cond_layers[index], 2, axis=-1)
            half_tanh = tf.nn.tanh(half_tanh)
            half_sigmoid = tf.nn.sigmoid(half_sigmoid)

            activated = half_tanh * half_sigmoid
            res_skip_activation = self.res_skip_layers[index](activated)

            if index < (self.n_layers - 1):
                res_skip_activation_0, res_skip_activation_1 = tf.split(res_skip_activation, 2, axis=-1)
                started = res_skip_activation_0 + started
                skip_activation = res_skip_activation_1
            else:
                skip_activation = res_skip_activation

            if index == 0:
                output = skip_activation
            else:
                output = skip_activation + output

        output = self.end(output)
        return output

    def build(self, input_shape):
        self._built_input_shape = input_shape
        super().build(input_shape)

    @property
    def output_shape(self):
        o_shape = self._built_input_shape
        if self._built_input_shape is not None:
            return o_shape[0][0], o_shape[0][1], self.n_out_channels
        return o_shape

    def get_config(self):
        config = super().get_config()
        config.update(n_in_channels=self.n_in_channels)
        config.update(n_channels=self.n_channels)
        config.update(n_out_channels=self.n_out_channels)
        config.update(n_layers=self.n_layers)
        config.update(kernel_size=self.kernel_size)
        config.update(dilation_rate_step=self.dilation_rate_step)
        config.update(max_log2_dilation_rate=self.max_log2_dilation_rate)
        config.update(use_weight_norm=self.use_weight_norm)
        return config


# ## WaveNet And Affine Coupling
# This block is a convenience block which has been defined to make it more straightforward to implement the WaveGlow model using the keras functional API.
# Note that affine coupling is the choice made in the original implementation of WaveGlow, but other choices are possible.
class WaveNetAffineBlock(layers.Layer):
    """
    Wavenet + Affine Layer
    Convenience block to provide a tidy model definition
    """

    def __init__(self, n_in_channels, n_channels=256,
                 n_layers=12, kernel_size=3, dilation_rate_step=1, max_log2_dilation_rate=None,
                 wavenet_n_out_channels = None,
                 use_weight_norm = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_in_channels = n_in_channels
        self.kernel_size = kernel_size
        self.dilation_rate_step = dilation_rate_step
        self.max_log2_dilation_rate = max_log2_dilation_rate
        self.wavenet_n_out_channels = wavenet_n_out_channels
        self.use_weight_norm = use_weight_norm
        self.wavenet = WaveNetNvidia(n_in_channels=n_in_channels,
                                     n_channels=n_channels,
                                     n_layers=n_layers,
                                     kernel_size=kernel_size,
                                     dilation_rate_step=dilation_rate_step,
                                     max_log2_dilation_rate=max_log2_dilation_rate,
                                     n_out_channels=wavenet_n_out_channels,
                                     use_weight_norm=self.use_weight_norm,
                                     dtype=self.dtype)

        self.affine_coupling = AffineCoupling(dtype=self.dtype)

        self._built_input_shape = None

    @property
    def output_shape(self):
        out_shape = None
        if self._built_input_shape is not None:
            out_shape = self.wavenet.output_shape
            out_shape = self.affine_coupling.output_shape
        return out_shape

    def build(self, input_shape):
        self._built_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        training should be set to false to inverse affine layer
        """
        audio_cond = None
        if len(inputs) == 3:
            audio, audio_cond, spect = inputs
        else:
            audio, spect = inputs

        audio_0, audio_1 = tf.split(audio, 2, axis=-1)

        if audio_cond is not None:
            audio_0pc = tf.concat((audio_0, audio_cond), axis=-1)
            wavenet_output = self.wavenet((audio_0pc, spect))
        else:
            wavenet_output = self.wavenet((audio_0, spect))

        audio_1 = self.affine_coupling(
            (audio_1, wavenet_output), training=training)

        audio = tf.concat([audio_0, audio_1], axis=-1)
        return audio

    def get_config(self):
        config = super(WaveNetAffineBlock, self).get_config()
        config.update(n_in_channels=self.n_in_channels)
        config.update(n_channels=self.n_channels)
        config.update(n_layers=self.n_layers)
        config.update(wavenet_n_out_channels=self.wavenet_n_out_channels)
        config.update(kernel_size=self.kernel_size)
        config.update(dilation_rate_step=self.dilation_rate_step)
        config.update(max_log2_dilation_rate=self.max_log2_dilation_rate)
        config.update(use_weight_norm=self.use_weight_norm)
        return config

