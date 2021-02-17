import tensorflow as tf
from debugprint import print_debug
import os
from fileio.iovar import load_var
import numpy as np


def weighted_binary_cross_entropy_fn(y_true, y_pred_logit, weights):
    """
    Weighted binary cross entropy with logit
    Parameters
    ----------
    y_true : true label
    y_pred_logit : predicted logit
    weights : two values array or dictionary with weights for class 0 and for class 1

    Returns
    -------
    loss result : tf float
    """
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    weights_v = tf.where(tf.equal(y_true, 1), weights[1], weights[0])
    loss = bce(y_true, y_pred_logit)
    return tf.reduce_mean(tf.multiply(loss, weights_v))


class GParrotLoss():
    def __init__(self, hparams):
        super(GParrotLoss, self).__init__()

        # loss weights
        self.gauto_w = hparams.gender_autoencoder_loss_w  # gender auto encoder
        self.gadv_w = hparams.gender_adversarial_loss_w  # gender adversarial from classifier
        self.gender_autoencoder_error_type = hparams.gender_autoencoder_error_type

        # use discriminator if True
        self.gender_use_adversarial_discriminator = hparams.gender_use_adversarial_discriminator

        # FOR STANDARDIZATION
        spkencoder_standardize_file = os.path.join(hparams.database_root_dir,
                                                   hparams.mel_mean_std_speaker_encoder)
        data_dict = load_var(spkencoder_standardize_file)
        self.spk_standardize_std = tf.convert_to_tensor(np.tile(data_dict['std_mell'].T,  # input => 128 x 1
                                                                (hparams.batch_size, 1)),
                                                        dtype=tf.float32)  # resized -> batch_size x 128

    def compute_loss(self, model_outputs, gender_target, speaker_target, eps=1e-5):

        speaker_logit_from_mel, speaker_embedding, speaker_embedding_out, gender_logit_from_gdisc = model_outputs

        # classical speaker encoder loss/accuracy - same than base code
        speaker_encoder_loss = tf.constant(0., dtype=tf.dtypes.float32)  # tf.reduce_mean(loss)
        predicted_speaker = tf.cast(tf.math.argmax(speaker_logit_from_mel, axis=1), dtype=tf.int16)
        speaker_encoder_acc = tf.reduce_sum(tf.cast((predicted_speaker == speaker_target), tf.float32)) / speaker_target.shape[0]

        # gender autoencoder loss
        # gender_autoencoder_destandardized_loss = reconstruction loss on de-standardized data
        if self.gender_autoencoder_error_type == 'mse':
            gender_autoencoder_loss = tf.reduce_mean(tf.abs(speaker_embedding-speaker_embedding_out)**2)
            gender_autoencoder_destandardized_loss = tf.reduce_mean((self.spk_standardize_std *
                                                                   tf.abs(speaker_embedding - speaker_embedding_out))**2)
        elif self.gender_autoencoder_error_type == 'mae':
            gender_autoencoder_loss = tf.reduce_mean(tf.abs(speaker_embedding - speaker_embedding_out))
            gender_autoencoder_destandardized_loss = tf.reduce_mean(self.spk_standardize_std *
                                                                   tf.abs(speaker_embedding - speaker_embedding_out))
        elif self.gender_autoencoder_error_type == 'dotproduct':
            dotprod = tf.reduce_sum(speaker_embedding*speaker_embedding_out, axis=1)
            norm_in = tf.norm(speaker_embedding, ord='euclidean', axis=1)
            norm_out = tf.norm(speaker_embedding_out, ord='euclidean', axis=1)
            gender_autoencoder_loss = 1.-tf.reduce_mean(dotprod/(norm_in*norm_out))
            gender_autoencoder_destandardized_loss = gender_autoencoder_loss
        else:
            import sys
            sys.exit('unknown gender autoencoder loss')

        # gender classification loss/accuracy
        gender_target_flatten = tf.keras.backend.flatten(gender_target)

        if self.gender_use_adversarial_discriminator:
            gender_logit_from_gdisc_flatten = tf.keras.backend.flatten(gender_logit_from_gdisc)
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True) \
                (gender_target_flatten, gender_logit_from_gdisc_flatten)
            gender_classification_loss = tf.reduce_mean(loss)
            # gender_classification_loss = weighted_binary_cross_entropy_fn(gender_target_flatten,
            #                                                         gender_logit_from_gdisc_flatten, self.bce_weights)
            predicted_gender = tf.cast(gender_logit_from_gdisc_flatten > 0., dtype=tf.int16)
            gender_classification_acc = tf.reduce_mean(tf.cast(predicted_gender == gender_target_flatten, dtype=tf.float32))
        else:
            gender_classification_loss = tf.constant(0., dtype=tf.dtypes.float32)
            gender_classification_acc = tf.constant(0., dtype=tf.dtypes.float32)

        # bce adversarial loss
        if self.gender_use_adversarial_discriminator:
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True) \
                (1-gender_target_flatten, gender_logit_from_gdisc_flatten)
            gender_adv_loss = tf.reduce_mean(loss)
        else:
            gender_adv_loss = tf.constant(0., dtype=tf.dtypes.float32)

        loss_list = [speaker_encoder_loss, gender_autoencoder_loss, gender_classification_loss, gender_adv_loss,
                     gender_autoencoder_destandardized_loss]
        accuracy_list = [speaker_encoder_acc, gender_classification_acc]

        combined_loss1 = self.gauto_w*gender_autoencoder_loss+self.gadv_w*gender_adv_loss
        combined_loss2 = gender_classification_loss

        return loss_list, accuracy_list, combined_loss1, combined_loss2
