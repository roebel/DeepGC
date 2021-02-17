import tensorflow as tf
from .layersgender import GEncoder, GDecoder, Gdiscriminator, GPreDiscriminator, SpeakerEncoder
from debugprint import print_debug
import os
from fileio.iovar import load_var
import numpy as np
import tensorflow_probability as tfp


class GParrot(tf.keras.Model):
    def __init__(self, hparams):
        super(GParrot, self).__init__()

        self.speaker_encoder = SpeakerEncoder(hparams)
        self.gencoder = GEncoder(hparams)
        self.gdecoder = GDecoder(hparams)
        self.gdicriminator = Gdiscriminator(hparams)
        self.gprediscriminator = GPreDiscriminator(hparams)

        # parameters
        # use gender discriminator if True
        self.gender_use_adversarial_discriminator = hparams.gender_use_adversarial_discriminator
        self.speaker_embedding_dim = hparams.speaker_embedding_dim  # technical need
        self.fine_tune = hparams.fine_tune
        self.dummy_constant = tf.constant([0], dtype=tf.dtypes.float32)
        self.soft_gender_id = hparams.soft_gender_id

        self.speaker_encoder_data_standardize = hparams.speaker_encoder_data_standardize
        if self.speaker_encoder_data_standardize:
            spkencoder_standardize_file = os.path.join(hparams.database_root_dir,
                                                       hparams.mel_mean_std_speaker_encoder)
            data_dict = load_var(spkencoder_standardize_file)
            self.spk_standardize_mean = tf.convert_to_tensor(np.tile(data_dict['mean_mell'].T,  # input => 128 x 1
                                                                     (hparams.batch_size, 1)),
                                                             dtype=tf.float32)  # resized -> batch_size x 128
            self.spk_standardize_std = tf.convert_to_tensor(np.tile(data_dict['std_mell'].T,  # input => 128 x 1
                                                                            (hparams.batch_size, 1)),
                                                                    dtype=tf.float32)  # resized -> batch_size x 128
            self.spk_standardize_std += tf.keras.backend.epsilon()  # add epsilon

    @property
    def main_trainable_weights(self):
        weights = []
        weights.extend(self.gencoder.trainable_weights)
        weights.extend(self.gdecoder.trainable_weights)
        return weights

    @property
    def sc_trainable_weights(self):
        # gender classifier
        return self.gdicriminator.trainable_weights

    def get_config(self):
        return {"fine_tune": self.fine_tune},

    def build(self, input_shape):

        mel_inputs_shape = input_shape[0]
        mel_lengths_shape = input_shape[2]

        # transpose mel input shape
        mel_inputs_shape = (mel_inputs_shape[0], mel_inputs_shape[2], mel_inputs_shape[1])

        self.speaker_encoder.build([mel_inputs_shape, mel_lengths_shape])
        _, speaker_embedding_shape = self.speaker_encoder.compute_output_shape([mel_inputs_shape, mel_lengths_shape])

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
             swap_gender=False, predict_gender_only=False, output_hidden_code=False):
        """

        Parameters
        ----------
        inputs: input list of tensors
        use_text: bool, unused here
        training: training/test!
        do_voice_conversion: if True, use mel_reference (in inputs list) as speaker encoder input
        use_true_gender_id: if False use pedicted gender by the pre-gender discriminator
        swap_gender: if True, gender id -> 1 - gender id (female -> male and conversely)
        predict_gender_only: if True, outputs gender (from speaker encoder and at the output of the auto-encoder)
        output_hidden_code: if true, outputs gender autoencoder latent code

        Returns
        -------

        """
        mel_inputs, text_inputs, mel_lengths, expand_mat_padded, mat_onehot_padded, speaker_id, mel_reference = inputs

        mel_inputs = tf.transpose(mel_inputs, (0, 2, 1))

        # speaker encoders
        if do_voice_conversion:
            speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(mel_reference, training=False)
        else:
            speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([mel_inputs, mel_lengths],
                                                                             training=False)
        stopgrad_speaker_embedding = tf.stop_gradient(speaker_embedding)

        # standardize output of speaker encoder
        if self.speaker_encoder_data_standardize:
            stopgrad_speaker_embedding = (stopgrad_speaker_embedding - self.spk_standardize_mean) / self.spk_standardize_std

        if use_true_gender_id:
            # true gender id
            gender_id = tf.expand_dims(tf.cast(mat_onehot_padded, dtype=tf.float32), axis=1)
        else:
            # normal flow
            gender_id = tf.sigmoid(self.gprediscriminator(stopgrad_speaker_embedding, training=False))

        if swap_gender:
            gender_id = 1.-gender_id

        gcode = self.gencoder(stopgrad_speaker_embedding, training=training)
        if output_hidden_code:
            return gcode

        gcode_full = tf.concat([gcode, gender_id], axis=-1)
        speaker_embedding_out = self.gdecoder(gcode_full, training=training)

        if predict_gender_only:
            gender_pred_out = self.gprediscriminator(speaker_embedding_out) > 0.
            gender_pred_in = self.gprediscriminator(stopgrad_speaker_embedding) > 0.
            return gender_pred_in, gender_pred_out

        if self.gender_use_adversarial_discriminator:
            gender_logit_from_gdisc = self.gdicriminator(gcode, training=training)
        else:
            gender_logit_from_gdisc = tf.constant(0., dtype=tf.dtypes.float32)

        return (speaker_logit_from_mel, stopgrad_speaker_embedding, speaker_embedding_out, gender_logit_from_gdisc)


class SexGParrot(tf.keras.Model):
    def __init__(self, hparams):
        super(SexGParrot, self).__init__()

        self.gprediscriminator = GPreDiscriminator(hparams)
        self.speaker_encoder = SpeakerEncoder(hparams)

        # parameters
        self.speaker_embedding_dim = hparams.speaker_embedding_dim  # technical need
        self.dummy_constant = tf.constant([0], dtype=tf.dtypes.float32)

        self.speaker_encoder_data_standardize = hparams.speaker_encoder_data_standardize
        if self.speaker_encoder_data_standardize:
            spkencoder_standardize_file = os.path.join(hparams.database_root_dir, hparams.mel_mean_std_speaker_encoder)
            data_dict = load_var(spkencoder_standardize_file)
            self.spk_standardize_mean = tf.convert_to_tensor(np.tile(data_dict['mean_mell'].T,  # input => 128 x 1
                                                                     (hparams.batch_size, 1)),
                                                             dtype=tf.float32)  # resized -> batch_size x 128
            self.spk_standardize_std = tf.convert_to_tensor(np.tile(data_dict['std_mell'].T,  # input => 128 x 1
                                                            (hparams.batch_size, 1)),
                                                            dtype=tf.float32)  # resized -> batch_size x 128
            self.spk_standardize_std += tf.keras.backend.epsilon()  # add epsilon

        # add noise to speaker embedding to train "dummy" discriminator
        self.sexprediscriminator_addnoise = hparams.sexprediscriminator_addnoise
        if self.sexprediscriminator_addnoise:
            self.noisescale = hparams.noisescale
            self.normal = tfp.distributions.Normal(loc=0., scale=self.noisescale)

    @property
    def gprec_trainable_weights(self):
        # pre-gender classifier
        return self.gprediscriminator.trainable_weights

    def build(self, input_shape):

        mel_inputs_shape = input_shape[0]
        mel_lengths_shape = input_shape[2]

        # transpose mel input shape
        mel_inputs_shape = (mel_inputs_shape[0], mel_inputs_shape[2], mel_inputs_shape[1])

        self.speaker_encoder.build([mel_inputs_shape, mel_lengths_shape])
        _, speaker_embedding_shape = self.speaker_encoder.compute_output_shape([mel_inputs_shape, mel_lengths_shape])

        self.gprediscriminator.build(speaker_embedding_shape)
        self.built = True

    def parse_batch(self, batch, use_gpu=True):

        (text_input_padded_text_level, text_input_padded, mel_padded, mat_onehot_padded,
         expand_mat_padded, speaker_id, text_lengths, mel_lengths,
         stop_token_padded) = batch

        return ((text_input_padded, mel_padded, mat_onehot_padded, expand_mat_padded,
                 text_lengths, mel_lengths),
                (text_input_padded_text_level, text_input_padded, mel_padded, speaker_id, stop_token_padded))

    def call(self, inputs, use_text=None, training=None):
        """

        Parameters
        ----------
        inputs: input list of tensors
        use_text: bool, unused here
        training: training/test!


        Returns
        -------

        """
        mel_inputs, text_inputs, mel_lengths, expand_mat_padded, mat_onehot_padded, speaker_id, mel_reference = inputs

        mel_inputs = tf.transpose(mel_inputs, (0, 2, 1))

        # speaker encoder
        speaker_logit_from_mel, speaker_embedding = self.speaker_encoder([mel_inputs, mel_lengths], training=False)
        stopgrad_speaker_embedding = tf.stop_gradient(speaker_embedding)

        # add noise to decrease performances
        if training and self.sexprediscriminator_addnoise:
            stopgrad_speaker_embedding += self.normal.sample(sample_shape=stopgrad_speaker_embedding.shape)

        # standardize output of speaker encoder
        if self.speaker_encoder_data_standardize:
            stopgrad_speaker_embedding = (stopgrad_speaker_embedding - self.spk_standardize_mean) / self.spk_standardize_std

        gender_logit_from_gdisc = self.gprediscriminator(stopgrad_speaker_embedding, training)
        return (speaker_logit_from_mel, self.dummy_constant, self.dummy_constant, gender_logit_from_gdisc)
