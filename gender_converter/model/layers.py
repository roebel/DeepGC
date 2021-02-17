import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, LSTM, Bidirectional, Dropout
BatchNormalization._USE_V2_BEHAVIOR = False  # solves problem with custom 'build' method
from tensorflow.keras import Sequential
# from .basic_layers import ConvNorm, LinearNorm
from .utils import get_mask_from_lengths
from tensorflow.keras.layers import Dense, Conv1D, Activation
import numpy as np
from debugprint import print_debug


class TextEncoder(Layer):
    """Text Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(TextEncoder, self).__init__()

        assert hparams.use_text_encoder_dilatation is False
        batchNormalizationAxis = 2
        self.convolutions = []

        for i in range(hparams.encoder_n_convolutions):
            # conv = Sequential(name='textencoder' + str(i))
            self.convolutions.append(Conv1D(filters=hparams.encoder_embedding_dim, kernel_size=hparams.encoder_kernel_size,
                            strides=1,
                            padding="same", data_format='channels_last', dilation_rate=1, use_bias=True))
            self.convolutions.append(BatchNormalization(axis=batchNormalizationAxis))
            self.convolutions.append(tf.keras.layers.Activation('relu'))
            self.convolutions.append(Dropout(rate=hparams.text_encoder_dropout))

        # layer parameters
        self.batchNormalizationAxis = batchNormalizationAxis
        self.encoder_n_convolutions = hparams.encoder_n_convolutions
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.encoder_kernel_size = hparams.encoder_kernel_size
        self.text_encoder_dropout = hparams.text_encoder_dropout

        self.lstm = Bidirectional(LSTM(units=hparams.encoder_embedding_dim//2, return_sequences=True))  # ,
        self.projection = Dense(hparams.encoder_embedding_dim, activation='tanh')

    def get_config(self):
        return {"batchNormalizationAxis": self.batchNormalizationAxis,
                "encoder_n_convolutions": self.encoder_n_convolutions,
                "encoder_embedding_dim": self.encoder_embedding_dim,
                "encoder_kernel_size": self.encoder_kernel_size,
                "text_encoder_dropout": self.text_encoder_dropout}

    def call(self, inputs, training=None):
        inputs, mel_lengths = inputs
        x = inputs
        for conv in self.convolutions:
            x = conv(x, training=training)
        mask = get_mask_from_lengths(lengths=mel_lengths)
        outputs = self.lstm(x, mask=mask)
        outputs = self.projection(outputs)

        return outputs

    def build(self, input_shape):
        current_shape = input_shape[0]
        with tf.name_scope(self.name):
            for ss in self.convolutions:
                ss.build(current_shape)
                current_shape = ss.compute_output_shape(current_shape)
            self.lstm.build(current_shape)
            current_shape = self.lstm.compute_output_shape(current_shape)
            self.projection.build(current_shape)
        super().build(input_shape)


class MergeNet(Layer):
    '''
    one layer bi-lstm
    '''

    def __init__(self, hparams):
        super(MergeNet, self).__init__()
        self.lstm = Bidirectional(LSTM(input_shape=(None, hparams.encoder_embedding_dim),
                                       units=hparams.encoder_embedding_dim//2, return_sequences=True))
        self.encoder_embedding_dim = hparams.encoder_embedding_dim

    def get_config(self):
        return {"encoder_embedding_dim": self.encoder_embedding_dim}

    def call(self, inputs, training=None):

        inputs, mel_lengths = inputs
        x = inputs
        mask = get_mask_from_lengths(lengths=mel_lengths)
        outputs = self.lstm(x, mask=mask)

        return outputs

    def build(self, input_shape):
        # print_debug(input_shape)
        with tf.name_scope(self.name):
            current_shape = input_shape[0]
            self.lstm.build(current_shape)
        super().build(input_shape)


class MelEncoder(Layer):
    '''
    -  Simple 2 layer bidirectional LSTM with projection
    '''

    def __init__(self, hparams):
        super(MelEncoder, self).__init__()
        self.lstm = []  # Sequential(name='melencoder')
        self.lstm.append(Bidirectional(LSTM(input_shape=(None, hparams.n_mel_channels),
                                         units=hparams.mel_encoder_hidden_dim//2, return_sequences=True,
                                         dropout=hparams.mel_encoder_dropout)))
        self.lstm.append(Bidirectional(LSTM(units=hparams.mel_encoder_hidden_dim//2, return_sequences=True,
                                         dropout=hparams.mel_encoder_dropout)))

        self.projection = Dense(hparams.mel_embedding_dim, activation='tanh')
        self.dropout = Dropout(rate=hparams.mel_encoder_dropout)
        self.project_to_n_symbols = Dense(hparams.n_symbols)
        self.n_symbols = hparams.n_symbols

        # parameters
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_encoder_hidden_dim = hparams.mel_encoder_hidden_dim
        self.mel_encoder_dropout = hparams.mel_encoder_dropout
        self.mel_embedding_dim = hparams.mel_embedding_dim

    def get_config(self):
        return {"n_symbols": self.n_symbols,
                "n_mel_channels": self.n_mel_channels,
                "mel_encoder_hidden_dim": self. mel_encoder_hidden_dim,
                "mel_encoder_dropout": self.mel_encoder_dropout,
                "mel_embedding_dim": self.mel_embedding_dim}

    def call(self, inputs, training=None):

        inputs, mel_lengths = inputs
        x = inputs
        mask = get_mask_from_lengths(lengths=mel_lengths)
        for ss in self.lstm:
            x = ss(x, mask=mask, training=training)
        outputs = self.projection(x)

        # predicted text labels
        # [batch_size, T, hparams.n_symbols]
        logit = self.project_to_n_symbols(self.dropout(outputs, training=training))

        return outputs, logit

    def build(self, input_shape):
        current_shape = input_shape[0]
        # self.lstm.build(current_shape)
        with tf.name_scope(self.name):
            for ss in self.lstm:
                ss.build(current_shape)
                current_shape = ss.compute_output_shape(current_shape)
            self.projection.build(current_shape)
            current_shape = self.projection.compute_output_shape(current_shape)
            self.project_to_n_symbols.build(current_shape)
        super().build(input_shape)


class FullSpeakerEncoder(Layer):
    def __init__(self, hparams):
        super().__init__()
        self.fine_tune = hparams.fine_tune
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.n_speakers = hparams.n_speakers

        # pre-train speaker embedding
        self.speaker_encoder = SpeakerEncoder(hparams)

        # fine-tune speaker embedding (binary class)
        self.sp_embedding = tf.keras.layers.Embedding(2, hparams.speaker_embedding_dim, trainable=self.fine_tune)

    def call(self, inputs, vc=False, training=None):
        if self.fine_tune:
            # unused output:
            speaker_logit_from_mel = tf.constant([0], dtype=tf.dtypes.float32)
            # speaker_id
            speaker_id = inputs[0]
            speaker_embedding = self.sp_embedding(speaker_id, training=training)
        else:
            speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(inputs, training=training)

        return speaker_logit_from_mel, speaker_embedding

    def build(self, input_shape):

        # print_debug(input_shape)
        current_shape = input_shape
        with tf.name_scope(self.name):
            self.sp_embedding.build(current_shape[1])
            self.speaker_encoder.build(current_shape)
            # input_shape = self.speaker_encoder.compute_output_shape(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (None, self.speaker_embedding_dim)

    def get_config(self):
        return {"fine_tune": self.fine_tune,
                "speaker_embedding_dim": self.speaker_embedding_dim,
                "n_speakers": self.n_speakers}

    def set_weights_emb(self, hparams):
        # set weights in fine tune
        a_embedding = np.load(hparams.a_embedding_path)
        a_embedding = np.mean(a_embedding, axis=0)

        b_embedding = np.load(hparams.b_embedding_path)
        b_embedding = np.mean(b_embedding, axis=0)

        cat_embedding = tf.convert_to_tensor(np.vstack((a_embedding, b_embedding)), dtype=tf.dtypes.float32)
        self.sp_embedding.set_weights([cat_embedding])


class SpeakerEncoder(Layer):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling
    '''
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.lstm = []  # Sequential(name='spkencoder')
        self.fine_tune = hparams.fine_tune
        self.lstm.append(Bidirectional(LSTM(units=hparams.speaker_encoder_hidden_dim//2, return_sequences=True,
                                         dropout=hparams.speaker_encoder_dropout,
                                            trainable=not self.fine_tune)))
        self.lstm.append(Bidirectional(LSTM(units=hparams.speaker_encoder_hidden_dim//2, return_sequences=True,
                                         dropout=hparams.speaker_encoder_dropout,
                                            trainable=not self.fine_tune)))
        self.projection1 = Dense(hparams.speaker_embedding_dim, activation='tanh',
                                 trainable=not self.fine_tune)
        self.projection2 = Dense(hparams.n_speakers,
                                 trainable=not self.fine_tune)

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
        outputs = self.projection1(outputs)
        # L2 normalizing #
        embeddings = outputs / tf.norm(outputs, axis=1, keepdims=True, ord='euclidean')

        logits = self.projection2(outputs)
        return logits, embeddings


class SpeakerClassifier(Layer):
    '''
    - n layer CNN + PROJECTION
    '''
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()

        batchNormalizationAxis = 2
        self.convolutions = []
        self.usetraining = []

        for i in range(hparams.SC_n_convolutions):
            self.convolutions.append(Conv1D(filters=hparams.SC_hidden_dim, kernel_size=hparams.SC_kernel_size,
                            strides=1,
                            padding="same", data_format='channels_last', dilation_rate=1, use_bias=True))
            self.usetraining.append(True)
            self.convolutions.append(BatchNormalization(axis=batchNormalizationAxis))
            self.usetraining.append(True)
            self.convolutions.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.usetraining.append(False)
            # conv.add(Dropout(rate=0.3))

        self.SC_n_convolutions = hparams.SC_n_convolutions
        self.SC_hidden_dim = hparams.SC_hidden_dim
        self.SC_kernel_size = hparams.SC_kernel_size
        self.batchNormalizationAxis = batchNormalizationAxis

        # projection => pre-train
        # projection_to_A => fine-tune
        # default configuration = pre-train
        self.fine_tune = hparams.fine_tune
        self.n_speakers = hparams.n_speakers
        self.projection = Dense(hparams.n_speakers, trainable=not self.fine_tune)
        self.projection_to_A = Dense(1, use_bias=False, trainable=self.fine_tune)

    def get_config(self):
        return {"SC_n_convolutions": self.SC_n_convolutions,
                "SC_hidden_dim": self.SC_hidden_dim,
                "SC_kernel_size": self.SC_kernel_size,
                "batchNormalizationAxis": self.batchNormalizationAxis,
                "fine_tune": self.fine_tune,
                "n_speakers": self.n_speakers}

    def build(self, input_shape):
        # https://stackoverflow.com/questions/59984492/naming-weights-of-a-layer-within-a-custom-layer
        # https: // www.tensorflow.org / api_docs / python / tf / name_scope

        # It seems that the outer name scope-related behaviours of Sequential and Dense are not consistent
        # in the eager mode.
        # This makes me thinking that this is a problem of Sequential that may still be isolated and fixed.
        # https://github.com/tensorflow/tensorflow/issues/36463

        current_shape = input_shape
        with tf.name_scope(self.name):
            for ss in self.convolutions:
                ss.build(current_shape)
                current_shape = ss.compute_output_shape(current_shape)
            # both projections are initialized
            self.projection.build(current_shape)
            self.projection_to_A.build(current_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.fine_tune:
            return [input_shape[0], 1]
        else:
            return [input_shape[0], self.n_speakers]

    def call(self, inputs, training=None):

        hidden = inputs
        for conv, val in zip(self.convolutions, self.usetraining):
            if val:
                hidden = conv(hidden, training=training)
            else:
                hidden = conv(hidden)

        if self.fine_tune:
            return self.projection_to_A(hidden)
        else:
            return self.projection(hidden)


class PostNet(Layer):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(PostNet, self).__init__()

        batchNormalizationAxis = 2
        self.convolutions = []

        for i in range(hparams.postnet_n_convolutions-1):
            self.convolutions.append(Conv1D(filters=hparams.postnet_dim, kernel_size=hparams.encoder_kernel_size,
                            strides=1,
                            padding="same", data_format='channels_last', dilation_rate=1, use_bias=True))
            self.convolutions.append(BatchNormalization(axis=batchNormalizationAxis))
            self.convolutions.append(Dropout(rate=hparams.postnet_dropout))
            self.convolutions.append(Activation("tanh"))
        # last layer outputs n_mel_channels
        self.convolutions.append(Conv1D(filters=hparams.n_mel_channels, kernel_size=hparams.encoder_kernel_size,
                    strides=1,
                    padding="same", data_format='channels_last', dilation_rate=1, use_bias=True))
        self.convolutions.append(BatchNormalization(axis=batchNormalizationAxis))
        self.convolutions.append(Dropout(rate=hparams.postnet_dropout))

        # parameters
        self.postnet_n_convolutions = hparams.postnet_n_convolutions
        self.postnet_dim = hparams.postnet_dim
        self.encoder_kernel_size = hparams.encoder_kernel_size
        self.batchNormalizationAxis = batchNormalizationAxis
        self.postnet_dropout = hparams.postnet_dropout
        self.n_mel_channels = hparams.n_mel_channels

    def get_config(self):
        return {"postnet_n_convolutions": self.postnet_n_convolutions,
                "postnet_dim": self.postnet_dim,
                "encoder_kernel_size": self.encoder_kernel_size,
                "batchNormalizationAxis": self.batchNormalizationAxis,
                "postnet_dropout": self.postnet_dropout,
                "n_mel_channels": self.n_mel_channels}

    def call(self, input, training=None):
        # input [B, T, mel_bins]

        x = input
        for conv in self.convolutions:
            x = conv(x, training=training)
        # resnet
        o = x + input
        return o

    def build(self, input_shape):
        current_shape = input_shape
        with tf.name_scope(self.name):
            for ss in self.convolutions:
                ss.build(current_shape)
                current_shape = ss.compute_output_shape(current_shape)
        super().build(input_shape)