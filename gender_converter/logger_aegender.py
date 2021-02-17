import random
from plotting_utils import plot_spectrogram_to_numpy, image_for_logger, plot_to_image
import numpy as np
import tensorflow as tf


class GParrotLogger():
    def __init__(self, logdir, ali_path='ali'):
        # super(ParrotLogger, self).__init__(logdir)

        self.writer = tf.summary.create_file_writer(logdir)

    def log_training(self, train_loss, loss_list, accuracy_list, grad_norm, learning_rate, duration, iteration):
        (speaker_encoder_loss, gender_autoencoder_loss, gender_classification_loss, gender_adv_loss,
         gender_autoencoder_destandardized_loss) = loss_list
        speaker_encoder_acc, gender_classification_acc = accuracy_list
        with self.writer.as_default():

            tf.summary.scalar("training.loss", train_loss, iteration)
            tf.summary.scalar("training.loss.spenc", speaker_encoder_loss, iteration)
            tf.summary.scalar("training.loss.gauto", gender_autoencoder_loss, iteration)
            tf.summary.scalar("training.loss.gautotrue", gender_autoencoder_destandardized_loss, iteration)
            tf.summary.scalar("training.loss.gcla", gender_classification_loss, iteration)
            tf.summary.scalar("training.loss.gadv", gender_adv_loss, iteration)

            tf.summary.scalar('training.acc.spenc', speaker_encoder_acc, iteration)
            tf.summary.scalar('training.acc.gcla', gender_classification_acc, iteration)

            tf.summary.scalar("grad.norm", grad_norm, iteration)
            tf.summary.scalar("learning.rate", learning_rate, iteration)
            tf.summary.scalar("duration", duration, iteration)
            self.writer.flush()

    def log_validation(self, val_loss, loss_list, accuracy_list, iteration):
        (speaker_encoder_loss, gender_autoencoder_loss, gender_classification_loss, gender_adv_loss,
         gender_autoencoder_destandardized_loss) = loss_list
        speaker_encoder_acc, gender_classification_acc = accuracy_list
        with self.writer.as_default():
            tf.summary.scalar("validation.loss", val_loss, iteration)
            tf.summary.scalar("validation.loss.spenc", speaker_encoder_loss, iteration)
            tf.summary.scalar("validation.loss.gauto", gender_autoencoder_loss, iteration)
            tf.summary.scalar("validation.loss.gautotrue", gender_autoencoder_destandardized_loss, iteration)
            tf.summary.scalar("validation.loss.gcla", gender_classification_loss, iteration)
            tf.summary.scalar("validation.loss.gadv", gender_adv_loss, iteration)

            tf.summary.scalar('validation.acc.spenc', speaker_encoder_acc, iteration)
            tf.summary.scalar('validation.acc.gcla', gender_classification_acc, iteration)
            self.writer.flush()
