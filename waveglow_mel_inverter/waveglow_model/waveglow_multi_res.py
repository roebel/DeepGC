#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import copy
import tensorflow as tf
from tensorflow.keras import layers

from .custom_layers import Inv1x1Conv, Inv1x1Conv_SVD, Inv1x1MM_PCA, Conv1DWeightNorm
from .custom_layers import WaveNetAffineBlock, WaveNetNvidia, SpectralUpsampling
from .training_utils import get_list_parameter


def split_layer_channels_split(audio, n_early_size, n_remaining_channels):
    output_chunk, audio = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=[n_early_size,
                                                                                  n_remaining_channels],
                                                           axis=2))(audio)
    return output_chunk, audio


class WaveGlow_MR(tf.keras.Model):
    """
    Waveglow implementation using the Invertible1x1Conv custom layer and
    the WaveNet custom block
    Likely change to have a hyper parameter dict
    The init function needs to be adjusted as we don't need to specify
    input dimension here as far as I understand the new 2.0 standards
    """

    def __init__(self, waveglow_config, training_config, preprocess_config,   **kwargs):
        super().__init__(dtype=training_config['ftype'], **kwargs)

        assert (waveglow_config['n_group'] % 2 == 0)
        self.waveglow_config = copy.deepcopy(waveglow_config)
        self.wavenet_config = self.waveglow_config['wavenet_config']
        self.svd_based_inv1x1_  = False
        self.preprocess_config = copy.deepcopy(preprocess_config)
        self.learned_permutation = True
        if ("learned_permutation" in  self.waveglow_config):
            self.learned_permutation = self.waveglow_config['learned_permutation']

        self.use_svd_for_permutation = False
        if self.learned_permutation and ("use_svd_for_permutation" in  self.waveglow_config):
            self.use_svd_for_permutation = self.waveglow_config['use_svd_for_permutation']

        self.training_config = copy.deepcopy(training_config)

        self.mel_processor = None
        self.n_flows = self.waveglow_config['n_flows']
        self.n_group = self.waveglow_config['n_group']
        self.sigma = self.waveglow_config['sigma']
        self.n_early_every = self.waveglow_config['n_early_every']
        self.mel_channels = self.preprocess_config['mel_channels']
        self.normalisation = self.training_config['train_batch_size'] * self.preprocess_config['segment_length']
        self.segment_length = self.preprocess_config['segment_length']
        # number of intermediate steps that need to be derived
        self.spect_hop_size = self.preprocess_config['hop_size']
        self.spect_steps = self.spect_hop_size // self.n_group
        if self.spect_steps * self.n_group !=  self.spect_hop_size:
            raise NotImplementedError(f"WaveGlow_MR::error::analysis hop size {self.spect_hop_size} not divisible by gouping factor {self.n_group}")
        self.waveNetAffineBlocks = []
        self.Inv1x1ConvLayers = []
        self.SpectMergers = []
        self.log_db_fac = 20*np.log10(2)/np.log(2)
        win_len = self.preprocess_config['fft_size']
        if 'win_size' in self.preprocess_config:
            win_len = self.preprocess_config['win_size']

        self.wrong_spect_conditioning = False
        spect_kernel_size = 3
        if ("wrong_spect_conditioning" in self.waveglow_config) and self.waveglow_config['wrong_spect_conditioning']:
            self.wrong_spect_conditioning = True
            spect_kernel_size = int(win_len//self.spect_hop_size)

        self.old_conditioning_layer = False
        if ("old_conditioning_layer" in self.waveglow_config) and self.waveglow_config['old_conditioning_layer']:
            self.old_conditioning_layer = True

        enable_weight_norm = True
        if 'use_weight_norm' in self.wavenet_config:
            raise RuntimeError("WaveGlow_MR::error:: use_weight_norm parameter did never have any effect and weight_norm "
                               "was always True even if set to False. "
                               "The new parameter now is named enable_weight_norm please update upur config file"
                               "renaming use_weigt_norm into enable_weight_norm and in case you have an old config "
                               "where use_wight_norm was Fals you need to set it to True because the old parameter value was never used.")
        if 'enable_weight_norm' in self.wavenet_config:
                enable_weight_norm = self.wavenet_config['enable_weight_norm']

        n_layers = get_list_parameter(self.wavenet_config['n_layers'], self.n_flows, self.n_early_every)
        n_channels = get_list_parameter(self.wavenet_config['n_channels'], self.n_flows, self.n_early_every)
        n_kernel_size = get_list_parameter(self.wavenet_config['kernel_size'], self.n_flows, self.n_early_every)

        # this is not what NVIDIA did, in tensorflow usamling just does repetition
        # NVIDIA does Conv1 mapping the 80 mel channels into 80 new channels
        # here I will simulate his effect by means of convolve1D multiplying the channels that will then be split over the
        # time axis to the multiple groups
        #self.upsampling = layers.UpSampling1D(size=self.upsampling_size,
        #                                      dtype=self.dtype)
        upsampling_filters = [self.spect_steps * self.mel_channels * 2]
        upsamling_activation_functions = [None]
        upsampling_kernel_sizes = [spect_kernel_size]
        self.inv1x1_kernels = []

        if "upsamling" in self.waveglow_config:
            upsampling_filters = self.waveglow_config["upsamling"]["filters"]
            upsamling_activation_functions = self.waveglow_config["upsamling"]["activation_functions"]
            upsampling_kernel_sizes = self.waveglow_config["upsamling"]["kernel_sizes"]
        elif "upsampling" in self.waveglow_config:
            upsampling_filters = self.waveglow_config["upsampling"]["filters"]
            upsamling_activation_functions = self.waveglow_config["upsampling"]["activation_functions"]
            upsampling_kernel_sizes = self.waveglow_config["upsampling"]["kernel_sizes"]

        if self.old_conditioning_layer:
            self.spect_up_to_depth = layers.Conv1D(filters=self.spect_steps * self.mel_channels * 2,
                                                       kernel_size=spect_kernel_size,
                                                       padding="SAME",
                                                       dtype=self.dtype, name="spect_upsampling")
        else:
            self.spect_up_to_depth = SpectralUpsampling(filters=upsampling_filters,
                                                        kernel_sizes=upsampling_kernel_sizes,
                                                        activation_functions=upsamling_activation_functions)

        self.whitening_layer = None
        if ("pre_white" in self.waveglow_config) and self.waveglow_config["pre_white"]:
            self.whitening_layer = Inv1x1MM_PCA(trainable=False)

        dilation_rate_step = 1
        if "dilation_rate_step" in self.wavenet_config :
            dilation_rate_step = self.wavenet_config['dilation_rate_step']

        n_dilation_rate_step = get_list_parameter(dilation_rate_step, self.n_flows, self.n_early_every)

        max_log2_dilation_rate = None
        if "max_log2_dilation_rate" in self.wavenet_config :
            max_log2_dilation_rate = self.wavenet_config['max_log2_dilation_rate']
        n_max_log2_dilation_rate = get_list_parameter(max_log2_dilation_rate, self.n_flows, self.n_early_every)

        self.output_facts = []
        n_half = self.n_group // 2
        for index in range(self.n_flows):
            if ((index % self.n_early_every == 0) and (index > 0)):
                if not self.output_facts:
                    self.output_facts.append(2)
                else:
                    self.output_facts.append(self.output_facts[-1] * 2)


                self.SpectMergers.append(Conv1DWeightNorm(filters=self.mel_channels * 2,
                                                          kernel_size=(spect_kernel_size-1) + self.output_facts[-1] + 1,
                                                          padding="SAME",
                                                          strides=self.output_facts[-1],
                                                          use_weight_norm=False,
                                                          dtype=self.dtype, name="spect_merger_{}".format(index)))


            if self.use_svd_for_permutation:
                self.Inv1x1ConvLayers.append(
                    Inv1x1Conv_SVD(filters=self.n_group,
                                   dtype=self.dtype,
                                   name="Inv1x1ConvSVD_{}".format(index)))
            else:
                self.Inv1x1ConvLayers.append(
                    Inv1x1Conv(filters=self.n_group,
                               dtype=self.dtype,
                               name="Inv1x1conv_{}".format(index)))

            self.waveNetAffineBlocks.append(
                WaveNetAffineBlock(n_in_channels=n_half,
                                   n_channels=n_channels[index],
                                   n_layers=n_layers[index],
                                   kernel_size=n_kernel_size[index],
                                   dtype=self.dtype,
                                   dilation_rate_step=n_dilation_rate_step[index],
                                   max_log2_dilation_rate=n_max_log2_dilation_rate[index],
                                   use_weight_norm=enable_weight_norm,
                                   name="waveNetAffineBlock_{}".format(index)))
        if not self.output_facts:
            self.output_facts.append(1)
        else:
            self.output_facts.append(self.output_facts[-1])

    def build_model(self):
        """
        this function constructs an input with batch size=1, time length =segment_length
        and calls the call method to construct all weights.
        Note that the keras build method cannot be used because the default action of thr call method is inference.
        """
        audio = np.zeros((1, self.segment_length, 1), np.float32)
        mel   = np.zeros((1, self.segment_length//self.spect_hop_size + 1, self.mel_channels), np.float32)
        self.__call__((audio, mel), training=True)

    def update_inverse(self):
        """
        sync backward kernel
        """
        if (not self.use_svd_for_permutation) and self.learned_permutation:
            for lay in self.Inv1x1ConvLayers:
                lay.update_inverse()

    def _create_upsampled_spec(self, spect, target_length):
        if spect.shape[1] * self.spect_hop_size < target_length:
            raise RuntimeError(f"spectrogram of length {spect.shape[1]}  "
                               f"does not fully cover the audio segment length={target_length}")

        # maps from batch x spect_time x mel_channels
        # into batch x spect_time x 1 x mel_channels * spect_steps
        tmp_spect = self.spect_up_to_depth(spect)
        upsampled_spect = tf.reshape(tmp_spect, (tmp_spect.shape[0],  tmp_spect.shape[1] * self.spect_steps, 2* self.mel_channels))

        if upsampled_spect.shape[1] < target_length//self.n_group:
            raise RuntimeError(f"after upsampling spectrogram of length {spect.shape[1]}  "
                               f"does not fully cover the audio segment length={target_length}")
        # crop to desired length
        upsampled_spect = upsampled_spect[:,:target_length//self.n_group,:]
        return upsampled_spect

    def call(self, inputs, training=None, *args, **kwargs):
        """
        Evaluate model against inputs
        
        if training is false simply return the output of the infer method,
        which effectively run through the layers backward and invert them.
        Otherwise run the network in the training "direction".
        """

        audio, spect = inputs

        if self.n_group * (audio.shape[1] // self.n_group) != audio.shape[1]:
            raise RuntimeError(f"WaveGlow_MR::error::audio length {audio.shape[1]} needs to be a multiple of n_group {self.n_group}")

        grp_audio = tf.reshape(audio, (audio.shape[0], -1, self.n_group))
        upsampled_spect = self._create_upsampled_spec(spect, audio.shape[1])
        output_latent = []
        n_half = self.n_group // 2
        used_upsampled_spect = upsampled_spect
        merger_index = 0

        if self.whitening_layer is not None:
            #print(f"before whitening: {(tf.transpose(grp_audio, (0,2,1)) @ grp_audio)/ grp_audio.shape[1]}")
            #print(f"before whitening: {tf.reduce_mean(tf.abs((tf.transpose(grp_audio, (0,2,1)) @ grp_audio)/ grp_audio.shape[1] - np.eye(self.n_group)))}")
            grp_audio = self.whitening_layer(grp_audio, training=True)
            # print(f"after whitening: {tf.reduce_mean(tf.abs((tf.transpose(grp_audio, (0,2,1)) @ grp_audio)/ grp_audio.shape[1]- np.eye(self.n_group)))}")

        for index in range(self.n_flows):
            if ((index % self.n_early_every == 0) and (index > 0)):
                output_chunk, grp_audio = split_layer_channels_split(audio=grp_audio, n_early_size=n_half,
                                                                 n_remaining_channels=n_half)
                output_latent.append(tf.reshape(output_chunk, (output_chunk.shape[0], -1,1)))
                # print(f"audio after split {index}: {audio.shape}")
                grp_audio = tf.reshape(grp_audio, (grp_audio.shape[0], -1, self.n_group))
                # print(f"audio after split + reshape: {grp_audio.shape}")
                used_upsampled_spect = self.SpectMergers[merger_index](upsampled_spect)
                merger_index += 1
                # output_latent.append(grp_audio[:, :, :self.n_early_size])
                # grp_audio = grp_audio[:,:,self.n_early_size:]

            # No need to output log_det_W or log_s as added as loss in custom
            # layers
            if self.inv1x1_kernels:
                grp_audio = self.Inv1x1ConvLayers[index]((grp_audio, self.inv1x1_kernels[index](used_upsampled_spect)))
            else:
                grp_audio = self.Inv1x1ConvLayers[index](grp_audio)

            # print(f"audio.shape {grp_audio.shape} mean abs audio {tf.reduce_mean(tf.abs(grp_audio))}", end="")
            grp_audio = self.waveNetAffineBlocks[index]((grp_audio, used_upsampled_spect),
                                                    training=True)
            #print(f"-> mean abs audio {tf.reduce_mean(tf.abs(grp_audio))}")

        output_latent.append(tf.reshape(grp_audio, (grp_audio.shape[0], -1, 1)))
        output_latent = tf.concat(output_latent, axis=1)
        if self.mel_processor and training:
            if False:
                # this should produce a minor error in
                infered_audio = self.infer(spect, z_in=output_latent)
                print("infer audio err:", audio.shape, infered_audio.shape, tf.reduce_mean(tf.abs(audio[:,:,0] - infered_audio)),
                      file=sys.stderr)
            else:
                infered_audio = self.infer(spect)

            mel_spec_inf = self.mel_processor(infered_audio)
            if self.mell_loss_ign_attn_db > 0:
                spect_min = (tf.reduce_max(tf.reduce_max(spect, keepdims=True, axis=1),
                                                        keepdims=True, axis=2)
                             - self.mell_loss_ign_attn_db / self.log_db_fac)
            else:
                spect_min =-100

            # if False:
            #     from matplotlib import pyplot as plt
            #
            #     print("infer spect shape", spect.shape, tf.reduce_max(spect[0]), mel_spec.shape, tf.reduce_max(mel_spec[0]))
            #     fige = plt.figure()
            #     axs = fige.add_subplot(211)
            #     axs.imshow(np.array(tf.maximum(spect[0], spect_min[0])).T)
            #     plt.title("ori")
            #     axr = fige.add_subplot(212, sharex=axs, sharey=axs)
            #     axr.imshow(np.array(tf.maximum(mel_spec[0,:spect.shape[1]], spect_min[0])).T)
            #     plt.title("synth")
            #     plt.show()
            self.add_loss(self.log_db_fac * tf.reduce_mean(tf.abs(tf.maximum(mel_spec_inf[:,:spect.shape[1]], spect_min)
                                                           - tf.maximum(spect, spect_min))))
        return output_latent

    def infer(self, spect, sigma=1.0, z_in =None, synth_length=0, sigma_wrap=None):
        """
        Push inputs through network in reverse direction.
        Two key aspects:
        Layers in reverse order.
        Layers are inverted through exposed training boolean.
        """

        #spect = layers.Reshape(
        #    target_shape=[63, self.mel_channels])(spect)
        if z_in is not None:
            synth_length= z_in.shape[1]
        else:
            synth_length = synth_length if synth_length else self.segment_length

        upsampled_spect = self._create_upsampled_spec(spect, synth_length)

        output_facts = list(self.output_facts)
        if z_in is None:
            audio = tf.random.normal(
                shape=[upsampled_spect.shape[0],
                       synth_length // (output_facts[-1] * self.n_group),
                       self.n_group],
                dtype=self.dtype)
            audio *= sigma
            if sigma_wrap is not None:
                audio = audio - sigma_wrap * tf.round(audio / sigma_wrap)
        else:
            z_in, audio = tf.split(z_in, [synth_length - synth_length//output_facts[-1],
                                          synth_length // output_facts[-1]], axis=1)

            # print(f"zin_shape {zinshaps} -> {z_in.shape} audio.shape {audio.shape}")
            audio = tf.reshape(audio, (audio.shape[0], -1, self.n_group))

        output_facts.pop()

        n_half = self.n_group // 2
        merger_index = len(self.SpectMergers) - 1
        if merger_index >= 0:
            used_upsampled_spect = self.SpectMergers[merger_index](upsampled_spect)
        else:
            used_upsampled_spect = upsampled_spect
        for index in reversed(range(self.n_flows)):

            #print(f"audio.shape {audio.shape} mean abs audio {tf.reduce_mean(tf.abs(audio))}", end="->")
            audio = self.waveNetAffineBlocks[index]((audio, used_upsampled_spect[:,:audio.shape[1],:]), training=False)
            #print(f" mean abs audio {tf.reduce_mean(tf.abs(audio))}")
            if self.inv1x1_kernels:
                audio = self.Inv1x1ConvLayers[index]((audio,
                                                      self.inv1x1_kernels[index](used_upsampled_spect[:,:audio.shape[1],:])),
                                                     training=False)
            else:
                audio = self.Inv1x1ConvLayers[index](audio, training=False)

            if ((index % self.n_early_every == 0) and (index > 0)):
                audio = tf.reshape(audio, (audio.shape[0], -1, n_half))
                if z_in is None:
                    z = tf.random.normal(
                        shape=[audio.shape[0],audio.shape[1], n_half],
                        dtype=self.dtype)
                    z *= sigma
                    if sigma_wrap is not None:
                        z = z - sigma_wrap * tf.round(z / sigma_wrap)
                else:
                    # print(f"output_facts: {output_facts}")
                    if z_in.shape[1]- synth_length//output_facts[-1]:
                        z_in, z = tf.split(z_in, [z_in.shape[1]- synth_length//output_facts[-1],
                                                  synth_length // output_facts[-1]], axis=1)
                    else:
                        z = z_in
                    # print(f"zin_shape {zinshaps} -> z shape {z.shape}")

                    z = tf.reshape(z, (z.shape[0],-1, n_half))
                audio = tf.concat([z, audio], axis=2)

                merger_index -= 1
                if merger_index >= 0:
                    used_upsampled_spect = self.SpectMergers[merger_index](upsampled_spect)
                else:
                    used_upsampled_spect = upsampled_spect

                output_facts.pop()

        if self.whitening_layer is not None:
            audio = self.whitening_layer(audio, training=False)

        audio = tf.reshape(audio, (audio.shape[0], -1))
        #self.add_loss(tf.reduce_mean(tf.abs(tf.exp(mel_spec[:, 2:spect.shape[1] - 2]) - tf.exp(spect[:, 2:-2]))))
        return audio


    def get_config(self):
        config = super(WaveGlow_MR, self).get_config()
        config.update(waveglow_config=self.waveglow_config)
        config.update(training_config=self.training_config)
        config.update(preprocess_config=self.preprocess_config)
        return config

    def format_loss(self, losses):
        return "".join(["{}:{:6.3f} ".format(ff, ll) for ff, ll in zip(["tot_loss", "LL_loss",
                                                                        "aff_loss", "ic_loss", "pca_loss", "mell_loss"], losses) if ll is not None])

    def total_loss(self, outputs):

        LL_loss = tf.reduce_sum(outputs * outputs) / (2 * self.sigma * self.sigma)

        affine_loss = tf.math.accumulate_n(
            [layer.losses[0] for layer in self.waveNetAffineBlocks])
        invconv_loss = tf.math.accumulate_n(
                [layer.losses[0] for layer in self.Inv1x1ConvLayers])


        LL_loss_normalized = LL_loss / self.normalisation
        affine_loss_normalized = affine_loss / self.normalisation
        # total_loss += (invconv_loss / self.n_group)
        invconv_loss_normalized = invconv_loss / self.normalisation
        total_loss = LL_loss_normalized + affine_loss_normalized + invconv_loss_normalized        


        whitening_loss_n = None
        if self.whitening_layer is not None:
            whitening_loss = self.whitening_layer.losses[0]
            whitening_loss_n = whitening_loss / self.normalisation
            total_loss += whitening_loss_n

        tf.summary.scalar(name='LL_loss_n',  data=LL_loss_normalized)
        tf.summary.scalar(name='affine_loss_n',  data=affine_loss_normalized)
        tf.summary.scalar(name='invconv_loss_n',  data=invconv_loss_normalized)

        if self.mel_processor is not None:
            mell_loss_n = self.losses[0]
            tf.summary.scalar(name='mell_loss_n', data=mell_loss_n)
            total_loss += self.mell_loss_weight * mell_loss_n
            tf.summary.scalar(name='total_loss_n', data=total_loss)
            return total_loss, LL_loss_normalized, affine_loss_normalized, invconv_loss_normalized, whitening_loss_n, mell_loss_n

        tf.summary.scalar(name='total_loss_n',  data=total_loss)
        return total_loss, LL_loss_normalized, affine_loss_normalized, invconv_loss_normalized, whitening_loss_n

