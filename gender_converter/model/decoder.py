import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, LSTM, Bidirectional, Dropout, Masking
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras import Sequential
from .utils import get_mask_from_lengths
from debugprint import print_debug


class Decoder(Layer):
    """
    2 layers bi-LSTM
    + projection to n_mel_bin
    """
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        # 2 layers bi-LSTM + projection to n_mel size

        # 2 layers bi-lstm
        self.lstm = []  # Sequential(name='decoderlstm')
        self.lstm.append(Bidirectional(LSTM(input_shape=(None, hparams.encoder_embedding_dim+hparams.speaker_embedding_dim),
                                         units=hparams.decoder_hidden_dim//2, return_sequences=True)))
        self.lstm.append(Bidirectional(LSTM(units=hparams.decoder_hidden_dim//2, return_sequences=True)))

        # parameters
        self.decoder_prenet = hparams.decoder_prenet
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.decoder_hidden_dim = hparams.decoder_hidden_dim
        self.prenet_dim = hparams.prenet_dim
        self.n_mel_channels = hparams.n_mel_channels

        if self.decoder_prenet:
            # pre-net
            # self.encoder = Sequential(name='decprenetencoder')
            self.encoder = []
            self.encoder.append(Dense(hparams.prenet_dim[0], activation='relu'))
            self.encoder.append(Dropout(rate=0.5))
            self.encoder.append(Dense(hparams.prenet_dim[1], activation='relu'))
            self.encoder.append(Dropout(rate=0.5))

            self.decoder = []
            self.decoder.append(Bidirectional(LSTM(units=hparams.decoder_hidden_dim//2, return_sequences=True)))
            self.decoder.append(Bidirectional(LSTM(units=hparams.decoder_hidden_dim//2, return_sequences=True)))

        # linear projection from LSTM output length to number of mel features
        self.proj_to_mel = Dense(hparams.n_mel_channels)

    def get_config(self):
        return {"decoder_prenet": self.decoder_prenet,
                "encoder_embedding_dim": self.encoder_embedding_dim,
                "speaker_embedding_dim": self.speaker_embedding_dim,
                "input_dim": self.encoder_embedding_dim+self.speaker_embedding_dim,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "prenet_dim": self.prenet_dim,
                "n_mel_channels": self.n_mel_channels}

    def call(self, inputs, training=None):

        inputs, mel_lengths = inputs
        x = inputs
        mask = get_mask_from_lengths(lengths=mel_lengths)
        for ss in self.lstm:
            x = ss(x, mask=mask)
        if self.decoder_prenet:
            for layer in self.encoder:
                x = layer(x, training=training)
            for layer in self.decoder:
                x = layer(x, training=training, mask=mask)
        outputs = x
        outputs = self.proj_to_mel(outputs)

        return outputs

    def build(self, input_shape):
        current_shape = input_shape[0]
        with tf.name_scope(self.name):
            for ss in self.lstm:
                ss.build(current_shape)
                current_shape = ss.compute_output_shape(current_shape)
            if self.decoder_prenet:
                for layer in self.encoder:
                    layer.build(current_shape)
                    current_shape = layer.compute_output_shape(current_shape)
                for layer in self.decoder:
                    layer.build(current_shape)
                    current_shape = layer.compute_output_shape(current_shape)
            self.proj_to_mel.build(current_shape)
        super().build(input_shape)
