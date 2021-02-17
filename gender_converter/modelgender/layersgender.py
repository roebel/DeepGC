import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Dropout, Dense, Activation
from tensorflow.keras.layers import Bidirectional, LSTM  # for speaker encoder
from .utils import get_mask_from_lengths
import numpy as np
from debugprint import print_debug

"""
Adapted from:
Adversarial Disentanglement of Speaker Representationfor Attribute-Driven Privacy Preservation
Paul-Gauthier Nóe, Mohammad MohammadAmini, Driss Matrouf,Titouan Parcollet, Jean-Francois Bonastre
https://hal.archives-ouvertes.fr/hal-03046920/document
with simplified networks!
"""


class GEncoder(Layer):
    """
        speaker encoder for gender layer (speaker space):
        Single Dense layer without activation
    """

    def __init__(self, hparams):
        super(GEncoder, self).__init__()
        self.dense = Dense(hparams.gender_latent_dim, trainable=True)
        self.gencoder_dim = hparams.gender_latent_dim

    def get_config(self):
        return {"gencoder_dim": self.gencoder_dim}

    def call(self, inputs, training=None):
        outputs = self.dense(inputs, training=training)
        return outputs

    def build(self, input_shape):
        current_shape = input_shape
        with tf.name_scope(self.name):
            self.dense.build(current_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [None, self.gencoder_dim]


class GDecoder(Layer):
    """speaker decoder for gender layer (after speaker encoder):
        simple Dense layer, with tanh activation
        1) removed final length normalization, multiplied output by a factor instead (> 1.)
    """

    def __init__(self, hparams):
        super(GDecoder, self).__init__()

        self.speaker_projection = Dense(hparams.speaker_embedding_dim, trainable=True, activation='tanh')
        # layer parameters
        self.gencoder_dim = hparams.speaker_embedding_dim
        self.gender_decoder_trainable_gain = hparams.gender_decoder_trainable_gain

    def get_config(self):
        return {"gdecoder_dim": self.gdecoder_dim}

    def call(self, inputs, training=None):
        """
        inputs is the concatenation of two tensors : 1) output of gencoder + 2) sex information (1D data)
        """
        outputs = self.gainaftertanh*self.speaker_projection(inputs, training=training)
        return outputs

    def build(self, input_shape):
        current_shape = input_shape
        with tf.name_scope(self.name):
            self.speaker_projection.build(current_shape)
            self.gainaftertanh = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(value=3.),
                                                 name='gain_after_tanh', trainable=self.gender_decoder_trainable_gain)
        super().build(input_shape)


class Gdiscriminator(Layer):
    """
        gender discriminator:
        Single Dense layer without activation
    """

    def __init__(self, hparams):
        super(Gdiscriminator, self).__init__()
        self.deeper_gender_discriminator_network = hparams.deeper_gender_discriminator_network
        self.use_gdiscriminator = hparams.gender_use_adversarial_discriminator
        if self.deeper_gender_discriminator_network:
            self.dense0 = Dense(hparams.gender_latent_dim//2, activation='relu', trainable=self.use_gdiscriminator)
        self.dense = Dense(1, trainable=self.use_gdiscriminator, use_bias=True, activation=None)

    def call(self, inputs, training=None):
        if self.deeper_gender_discriminator_network:
            x = self.dense0(inputs, training=training)
            outputs = self.dense(x, training=training)
        else:
            outputs = self.dense(inputs, training=training)
        return outputs

    def build(self, input_shape):
        current_shape = input_shape
        with tf.name_scope(self.name):
            if self.deeper_gender_discriminator_network:
                self.dense0.build(current_shape)
                current_shape = self.dense0.compute_output_shape(current_shape)
            self.dense.build(current_shape)
        super().build(input_shape)


class GPreDiscriminator(Layer):
    """
        pre-trained gender discriminator:
        a single Dense layer
    """

    def __init__(self, hparams):
        super(GPreDiscriminator, self).__init__()
        self.dense = Dense(1, trainable=hparams.gender_pretrain, use_bias=True, activation=None)

    def get_config(self):
        return {}

    def call(self, inputs, training=None):
        outputs = self.dense(inputs, training=training)
        return outputs

    def build(self, input_shape):
        current_shape = input_shape
        with tf.name_scope(self.name):
            self.dense.build(current_shape)
        super().build(input_shape)


class SpeakerEncoder(Layer):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling
    '''
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.lstm = []
        # self.fine_tune = hparams.fine_tune
        self.lstm.append(Bidirectional(LSTM(units=hparams.speaker_encoder_hidden_dim//2, return_sequences=True,
                                            dropout=hparams.speaker_encoder_dropout,
                                            trainable=False)))
        self.lstm.append(Bidirectional(LSTM(units=hparams.speaker_encoder_hidden_dim//2, return_sequences=True,
                                            dropout=hparams.speaker_encoder_dropout,
                                            trainable=False)))
        self.projection1 = Dense(hparams.speaker_embedding_dim, activation='tanh',
                                 trainable=False)
        self.projection2 = Dense(hparams.n_speakers,
                                 trainable=False)

        self.n_mel_channels = hparams.n_mel_channels
        self.speaker_encoder_hidden_dim = hparams.speaker_encoder_hidden_dim  # 256
        self.speaker_embedding_dim = hparams.speaker_embedding_dim  # 128
        self.speaker_encoder_dropout = hparams.speaker_encoder_dropout
        self.n_speakers = hparams.n_speakers

    def build(self, input_shape):
        # It seems that the outer name scope-related behaviours of Sequential and Dense are not consistent
        # in the eager mode.
        # This makes me thinking that this is a problem of Sequential that may still be isolated and fixed.
        # https://github.com/tensorflow/tensorflow/issues/36463
        current_shape = input_shape[0]
        with tf.name_scope(self.name):
            for ss in self.lstm:
                ss.build(current_shape)
                current_shape = ss.compute_output_shape(current_shape)
            current_shape = (current_shape[0], current_shape[2])
            self.projection1.build(current_shape)
            current_shape = self.projection1.compute_output_shape(current_shape)
            self.projection2.build(current_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [[None, self.n_speakers], [None, self.speaker_embedding_dim]]

    def get_config(self):
        return {"n_speakers": self.n_speakers,
                "speaker_embedding_dim": self.speaker_embedding_dim,
                "speaker_encoder_hidden_dim": self.speaker_encoder_hidden_dim,
                "speaker_encoder_dropout": self.speaker_encoder_dropout,
                "n_mel_channels": self.n_mel_channels}

    def call(self, inputs, training=None):
        inputs, mel_lengths = inputs
        mask = get_mask_from_lengths(lengths=mel_lengths)
        current_inputs = inputs
        for ss in self.lstm:
            current_inputs = ss(current_inputs, mask=mask, training=training)
        # global average pooling
        # mean pooling -> [batch_size, dim]
        outputs = tf.reduce_sum(current_inputs, axis=1) / tf.cast(mel_lengths, dtype=tf.float32)
        outputs = self.projection1(outputs, training=training)
        # L2 normalizing #
        embeddings = outputs / tf.norm(outputs, axis=1, keepdims=True, ord='euclidean')

        logits = self.projection2(outputs, training=training)
        return logits, embeddings
