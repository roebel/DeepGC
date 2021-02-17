import tensorflow as tf
from .layers import MelEncoder, MergeNet, SpeakerClassifier, TextEncoder, PostNet, FullSpeakerEncoder
from .layersgender import GEncoder, GDecoder, Gdiscriminator, GPreDiscriminator
from .decoder import Decoder
from debugprint import print_debug
from math import sqrt
from fileio.iovar import load_var
import os


class Parrot(tf.keras.Model):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        # embedding for text encoding
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding = tf.keras.layers.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim,
                                                   tf.keras.initializers.RandomUniform(minval=-val, maxval=val))

        self.merge_net = MergeNet(hparams)
        self.speaker_encoder = FullSpeakerEncoder(hparams)
        self.speaker_classifier = SpeakerClassifier(hparams)
        self.mel_encoder = MelEncoder(hparams)
        self.text_encoder = TextEncoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = PostNet(hparams)

        # parameters
        self.speaker_embedding_dim = hparams.speaker_embedding_dim  # technical need
        self.fine_tune = hparams.fine_tune
        self.dummy_constant = tf.constant([0], dtype=tf.dtypes.float32)

        # speaker classif logit from mel hidden can be at text rate or at frame rate
        self.spksclassif_at_mel_rate = hparams.spksclassif_at_mel_rate

    @property
    def main_trainable_weights(self):
        weights = []
        weights.extend(self.embedding.trainable_weights)
        weights.extend(self.merge_net.trainable_weights)
        weights.extend(self.speaker_encoder.trainable_weights)
        weights.extend(self.mel_encoder.trainable_weights)
        weights.extend(self.text_encoder.trainable_weights)
        weights.extend(self.decoder.trainable_weights)
        weights.extend(self.postnet.trainable_weights)
        return weights

    @property
    def sc_trainable_weights(self):
        # speaker classifier
        return self.speaker_classifier.trainable_weights

    def get_config(self):
        return {"fine_tune": self.fine_tune},

    def build(self, input_shape):

        mel_inputs_shape = input_shape[0]
        text_inputs_shape = input_shape[1]
        mel_lengths_shape = input_shape[2]

        # transpose mel input shape
        mel_inputs_shape = (mel_inputs_shape[0], mel_inputs_shape[2], mel_inputs_shape[1])

        self.speaker_encoder.build([mel_inputs_shape, mel_lengths_shape])
        speaker_embedding_shape = self.speaker_encoder.compute_output_shape([mel_inputs_shape, mel_lengths_shape])

        self.mel_encoder.build([mel_inputs_shape, mel_lengths_shape])
        mel_hidden_shape, _ = self.mel_encoder.compute_output_shape([mel_inputs_shape, mel_lengths_shape])

        self.embedding.build(text_inputs_shape)
        text_input_embedded_shape = self.embedding.compute_output_shape(text_inputs_shape)
        self.text_encoder.build([text_input_embedded_shape, mel_lengths_shape])

        # -> [B, text_len, n_speakers]
        mel_hidden_text_rate_shape = mel_hidden_shape  # .as_list()
        self.speaker_classifier.build(mel_hidden_text_rate_shape)

        self.merge_net.build([mel_hidden_shape, mel_lengths_shape])
        hidden_shape = self.merge_net.compute_output_shape([mel_hidden_shape, mel_lengths_shape])

        # # concat linguistic hidden tensor with speaker embed tensor
        hidden_shape_final = (None, None, hidden_shape[2] + speaker_embedding_shape[1])

        self.decoder.build([hidden_shape_final, mel_lengths_shape])
        mel_outputs_shape = self.decoder.compute_output_shape([hidden_shape_final, mel_lengths_shape])

        self.postnet.build(mel_outputs_shape)
        self.built = True

    def parse_batch(self, batch, use_gpu=True):

        (text_input_padded_text_level, text_input_padded, mel_padded, mat_onehot_padded,
         expand_mat_padded, speaker_id, text_lengths, mel_lengths,
         stop_token_padded) = batch

        return ((text_input_padded, mel_padded, mat_onehot_padded, expand_mat_padded,
                 text_lengths, mel_lengths),
                (text_input_padded_text_level, text_input_padded, mel_padded, speaker_id, stop_token_padded))

    def call(self, inputs, use_text=None, training=None, do_voice_conversion=None):

        mel_inputs, text_inputs, mel_lengths, expand_mat_padded, mat_onehot_padded, speaker_id, mel_reference = inputs

        mel_inputs = tf.transpose(mel_inputs, (0, 2, 1))

        # speaker encoders
        if self.fine_tune:
            if do_voice_conversion:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([mel_reference[0]],
                                                                                 training=training)
            else:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([speaker_id],
                                                                                 training=training)
        else:
            if do_voice_conversion:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(mel_reference, training=training)
            else:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([mel_inputs, mel_lengths],
                                                                                 training=training)

        mel_hidden, text_logit_from_mel_hidden = self.mel_encoder([mel_inputs, mel_lengths], training=training)

        text_input_embedded = self.embedding(text_inputs)  # .transpose(1, 2)
        text_hidden = self.text_encoder([text_input_embedded, mel_lengths], training=training)

        if self.spksclassif_at_mel_rate:  # classify speaker at mel rate
            mel_hidden_text_or_mel_rate = mel_hidden
        else:  # classify speaker from mel hidden at text rate
            # compress from mel rate to text rate (normalize by the phone durations)
            expand_normalization_vect = tf.reduce_sum(expand_mat_padded, axis=2, keepdims=True)+1e-10
            expand_mat_padded = expand_mat_padded/expand_normalization_vect
            # -> [B, text_len, n_speakers]
            mel_hidden_text_or_mel_rate = tf.linalg.matmul(expand_mat_padded, mel_hidden)

        speaker_logit_from_mel_hidden_text_or_mel_rate = self.speaker_classifier(mel_hidden_text_or_mel_rate, training)

        if use_text:
            hidden = self.merge_net([text_hidden, mel_lengths])
        else:
            hidden = self.merge_net([mel_hidden, mel_lengths])

        # concat linguistic hidden tensor with speaker embed tensor
        n_frames = tf.shape(hidden)[1]  # number of frames, needed for symbolic shapes, i.e. with Nones
        # https: // github.com / tensorflow / models / issues / 6245
        # n_frames = hidden.shape[1]  # number of frames
        if self.fine_tune:
            spkemb = tf.expand_dims(speaker_embedding, axis=1)
        else:
            # detach tensor here
            spkemb = tf.expand_dims(tf.stop_gradient(speaker_embedding), axis=1)
        spkemb = tf.tile(spkemb, [1, n_frames, 1])
        # spkemb.set_shape([None, None, self.speaker_embedding_dim])
        hidden = tf.concat([hidden, spkemb], axis=2)

        mel_outputs = self.decoder([hidden, mel_lengths], training=training)
        post_mel_outputs = self.postnet(mel_outputs, training=training)

        mel_outputs = tf.transpose(mel_outputs, (0, 2, 1))
        post_mel_outputs = tf.transpose(post_mel_outputs, (0, 2, 1))

        return (mel_outputs, post_mel_outputs, speaker_logit_from_mel, speaker_logit_from_mel_hidden_text_or_mel_rate,
                text_hidden, mel_hidden, mel_hidden_text_or_mel_rate, text_logit_from_mel_hidden)


class FullGenderParrot(tf.keras.Model):
    def __init__(self, hparams):
        super(FullGenderParrot, self).__init__()

        # embedding for text encoding
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        # VC LAYERS
        self.embedding = tf.keras.layers.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim,
                                                   tf.keras.initializers.RandomUniform(minval=-val, maxval=val))

        self.merge_net = MergeNet(hparams)
        self.speaker_encoder = FullSpeakerEncoder(hparams)
        self.speaker_classifier = SpeakerClassifier(hparams)
        self.mel_encoder = MelEncoder(hparams)
        self.text_encoder = TextEncoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = PostNet(hparams)

        # gender LAYERS
        self.gencoder = GEncoder(hparams)
        self.gdecoder = GDecoder(hparams)
        self.gdicriminator = Gdiscriminator(hparams)
        self.gprediscriminator = GPreDiscriminator(hparams)

        # parameters
        self.speaker_embedding_dim = hparams.speaker_embedding_dim  # technical need
        self.fine_tune = hparams.fine_tune
        self.dummy_constant = tf.constant([0], dtype=tf.dtypes.float32)

        # speaker classif logit from mel hidden can be at text rate or at frame rate
        self.spksclassif_at_mel_rate = hparams.spksclassif_at_mel_rate

        self.speaker_encoder_data_standardize = hparams.speaker_encoder_data_standardize

        if self.speaker_encoder_data_standardize:
            spkencoder_standardize_file = os.path.join(hparams.database_root_dir,
                                                       hparams.mel_mean_std_speaker_encoder)
            data_dict = load_var(spkencoder_standardize_file)
            self.spk_standardize_mean = tf.convert_to_tensor(data_dict['mean_mell'].T, dtype=tf.float32)
            # transposed -> 1 x 128
            self.spk_standardize_std = tf.convert_to_tensor(data_dict['std_mell'].T, dtype=tf.float32)
            # transposed -> 1 x 128
            self.spk_standardize_std += tf.keras.backend.epsilon()  # add epsilon

    def get_config(self):
        return {"fine_tune": self.fine_tune},

    @property
    def decoder_trainable_weights(self):
        weights = []
        weights.extend(self.decoder.trainable_weights)
        weights.extend(self.postnet.trainable_weights)
        return weights

    def build(self, input_shape):

        mel_inputs_shape = input_shape[0]
        text_inputs_shape = input_shape[1]
        mel_lengths_shape = input_shape[2]

        # transpose mel input shape
        mel_inputs_shape = (mel_inputs_shape[0], mel_inputs_shape[2], mel_inputs_shape[1])

        self.speaker_encoder.build([mel_inputs_shape, mel_lengths_shape])
        speaker_embedding_shape = self.speaker_encoder.compute_output_shape([mel_inputs_shape, mel_lengths_shape])

        self.mel_encoder.build([mel_inputs_shape, mel_lengths_shape])
        mel_hidden_shape, _ = self.mel_encoder.compute_output_shape([mel_inputs_shape, mel_lengths_shape])

        self.embedding.build(text_inputs_shape)
        text_input_embedded_shape = self.embedding.compute_output_shape(text_inputs_shape)
        self.text_encoder.build([text_input_embedded_shape, mel_lengths_shape])

        # -> [B, text_len, n_speakers]
        mel_hidden_text_rate_shape = mel_hidden_shape  # .as_list()
        self.speaker_classifier.build(mel_hidden_text_rate_shape)

        self.merge_net.build([mel_hidden_shape, mel_lengths_shape])
        hidden_shape = self.merge_net.compute_output_shape([mel_hidden_shape, mel_lengths_shape])

        # # concat linguistic hidden tensor with speaker embed tensor
        hidden_shape_final = (None, None, hidden_shape[2] + speaker_embedding_shape[1])

        self.decoder.build([hidden_shape_final, mel_lengths_shape])
        mel_outputs_shape = self.decoder.compute_output_shape([hidden_shape_final, mel_lengths_shape])

        self.postnet.build(mel_outputs_shape)

        # gender layers
        self.gencoder.build(speaker_embedding_shape)
        gencoder_shape = self.gencoder.compute_output_shape(speaker_embedding_shape)
        self.gdecoder.build([gencoder_shape[0], gencoder_shape[1]+1])

        self.gdicriminator.build(gencoder_shape)
        self.gprediscriminator.build(speaker_embedding_shape)

        self.built = True

    def parse_batch(self, batch, use_gpu=True):

        (text_input_padded_text_level, text_input_padded, mel_padded, mat_onehot_padded,
         expand_mat_padded, speaker_id, text_lengths, mel_lengths,
         stop_token_padded) = batch

        return ((text_input_padded, mel_padded, mat_onehot_padded, expand_mat_padded,
                 text_lengths, mel_lengths),
                (text_input_padded_text_level, text_input_padded, mel_padded, speaker_id, stop_token_padded))

    def call(self, inputs, use_text=None, training=None, do_voice_conversion=None, use_true_gender_id=False,
             swap_gender=False):

        mel_inputs, text_inputs, mel_lengths, expand_mat_padded, mat_onehot_padded, speaker_id, mel_reference = inputs

        mel_inputs = tf.transpose(mel_inputs, (0, 2, 1))

        # speaker encoders
        if self.fine_tune:
            if do_voice_conversion:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([mel_reference[0]],
                                                                                 training=training)
            else:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([speaker_id],
                                                                                 training=training)
        else:
            if do_voice_conversion:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(mel_reference, training=training)
            else:
                speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([mel_inputs, mel_lengths],
                                                                                 training=training)

        # speaker embedding gender stuff
        stopgrad_speaker_embedding = tf.stop_gradient(speaker_embedding)
        # standardize output of speaker encoder
        if self.speaker_encoder_data_standardize:
            stopgrad_speaker_embedding = (stopgrad_speaker_embedding - self.spk_standardize_mean) / self.spk_standardize_std
        if use_true_gender_id:
            # true gender id
            gender_id = tf.expand_dims(tf.cast(mat_onehot_padded, dtype=tf.float32), axis=1)
        else:
            # predicted gender id
            gender_id = tf.sigmoid(self.gprediscriminator(stopgrad_speaker_embedding, training=False))

        if swap_gender:
            gender_id = 1.-gender_id

        gcode = self.gencoder(stopgrad_speaker_embedding, training=training)
        gcode_full = tf.concat([gcode, gender_id], axis=-1)
        speaker_embedding_out = self.gdecoder(gcode_full, training=training)
        if self.speaker_encoder_data_standardize:
            speaker_embedding_out = speaker_embedding_out*self.spk_standardize_std + self.spk_standardize_mean

        # back to affairs
        mel_hidden, text_logit_from_mel_hidden = self.mel_encoder([mel_inputs, mel_lengths], training=training)

        text_input_embedded = self.embedding(text_inputs)  # .transpose(1, 2)
        text_hidden = self.text_encoder([text_input_embedded, mel_lengths], training=training)

        if self.spksclassif_at_mel_rate:  # classify speaker at mel rate
            mel_hidden_text_or_mel_rate = mel_hidden
        else:  # classify speaker from mel hidden at text rate
            # compress from mel rate to text rate (normalize by the phone durations)
            expand_normalization_vect = tf.reduce_sum(expand_mat_padded, axis=2, keepdims=True)+1e-10
            expand_mat_padded = expand_mat_padded/expand_normalization_vect
            # -> [B, text_len, n_speakers]
            mel_hidden_text_or_mel_rate = tf.linalg.matmul(expand_mat_padded, mel_hidden)

        speaker_logit_from_mel_hidden_text_or_mel_rate = self.speaker_classifier(mel_hidden_text_or_mel_rate, training)

        if use_text:
            hidden = self.merge_net([text_hidden, mel_lengths])
        else:
            hidden = self.merge_net([mel_hidden, mel_lengths])

        # concat linguistic hidden tensor with speaker embed tensor
        n_frames = tf.shape(hidden)[1]  # number of frames, needed for symbolic shapes, i.e. with Nones
        # https: // github.com / tensorflow / models / issues / 6245
        # n_frames = hidden.shape[1]  # number of frames
        if self.fine_tune:
            spkemb = tf.expand_dims(speaker_embedding, axis=1)
        else:
            # detach tensor here
            spkemb = tf.expand_dims(speaker_embedding_out, axis=1)

        spkemb = tf.tile(spkemb, [1, n_frames, 1])
        # spkemb.set_shape([None, None, self.speaker_embedding_dim])
        hidden = tf.concat([hidden, spkemb], axis=2)

        mel_outputs = self.decoder([hidden, mel_lengths], training=training)
        post_mel_outputs = self.postnet(mel_outputs, training=training)

        mel_outputs = tf.transpose(mel_outputs, (0, 2, 1))
        post_mel_outputs = tf.transpose(post_mel_outputs, (0, 2, 1))

        return (mel_outputs, post_mel_outputs, speaker_logit_from_mel, speaker_logit_from_mel_hidden_text_or_mel_rate,
                text_hidden, mel_hidden, mel_hidden_text_or_mel_rate, text_logit_from_mel_hidden)
