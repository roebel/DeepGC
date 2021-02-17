import random
from plotting_utils import plot_spectrogram_to_numpy, image_for_logger, plot_to_image
import numpy as np
import tensorflow as tf


class ParrotLogger():
    def __init__(self, logdir, ali_path='ali'):
        # super(ParrotLogger, self).__init__(logdir)

        self.writer = tf.summary.create_file_writer(logdir)

    def log_training(self, train_loss, loss_list, accuracy_list, grad_norm, learning_rate, duration, iteration, task):
        (recon_loss, recon_post_loss, speaker_encoder_loss, speaker_classification_loss, speaker_adversial_loss,
         text_classification_loss, contrast_loss) = loss_list
        speaker_encoder_acc, speaker_classification_acc, text_classification_acc = accuracy_list
        with self.writer.as_default():
            tf.summary.scalar("training.loss.%s" % task, train_loss, iteration-1)
            tf.summary.scalar("training.loss.%s" % task, train_loss, iteration)
            # tf.summary.scalar("training.loss.%s" % task, reduced_loss, iteration-1)
            # tf.summary.scalar("training.loss.%s" % task, reduced_loss, iteration)
            # separate reconstruction by task (TTS or VC)
            # ad-hoc trick to keep the same number of values in the reconstruction losses than in the other losses
            # tf.summary.scalar("training.loss.%s.recon" % task, 20*np.log10(reduced_losses[0]), iteration-1)
            # tf.summary.scalar("training.loss.%s.recon_post" % task, 20*np.log10(reduced_losses[1]), iteration-1)
            # reconstruction loss in dB
            tf.summary.scalar("training.loss.%s.recon_post" % task, 20*np.log10(recon_post_loss), iteration-1)
            tf.summary.scalar("training.loss.%s.recon_post" % task, 20*np.log10(recon_post_loss), iteration)
            # tf.summary.scalar("training.loss.%s.recon_post" % task, 20*np.log10(reduced_losses[1]), iteration)

            tf.summary.scalar("training.loss.contr", contrast_loss, iteration)
            tf.summary.scalar("training.loss.spenc", speaker_encoder_loss, iteration)
            tf.summary.scalar("training.loss.spcla", speaker_classification_loss, iteration)
            tf.summary.scalar("training.loss.texcl", text_classification_loss, iteration)
            tf.summary.scalar("training.loss.spadv", speaker_adversial_loss, iteration)

            tf.summary.scalar("grad.norm", grad_norm, iteration)
            tf.summary.scalar("learning.rate", learning_rate, iteration)
            tf.summary.scalar("duration", duration, iteration)

            # if iteration % 1000 == 0:
            #     tf.summary.image("input", plot_to_image(image_for_logger(y[0, :, :])), iteration)
            #     tf.summary.image("output", plot_to_image(image_for_logger(y_pred.numpy()[0, :, :])), iteration)
            tf.summary.scalar('training.acc.spenc', speaker_encoder_acc, iteration)
            tf.summary.scalar('training.acc.spcla', speaker_classification_acc, iteration)
            tf.summary.scalar('training.acc.texcl', text_classification_acc, iteration)
            #
            self.writer.flush()

    def log_validation(self, val_loss, loss_list, accuracy_list, y, y_pred, iteration, task):
        (recon_loss, recon_post_loss, speaker_encoder_loss, speaker_classification_loss, speaker_adversial_loss,
         text_classification_loss, contrast_loss) = loss_list
        speaker_encoder_acc, speaker_classification_acc, text_classification_acc = accuracy_list
        with self.writer.as_default():
            tf.summary.scalar("validation.loss.%s" % task, val_loss, iteration)
            # reconstruction loss in dB
            tf.summary.scalar("validation.loss.%s.recon_post" % task, 20*np.log10(recon_post_loss), iteration)
            tf.summary.scalar("validation.loss.%s.spenc" % task, speaker_encoder_loss, iteration)
            tf.summary.scalar("validation.loss.%s.spcla" % task, speaker_classification_loss, iteration)
            tf.summary.scalar("validation.loss.%s.spadv" % task, speaker_adversial_loss, iteration)
            tf.summary.scalar("validation.loss.%s.texcl" % task, text_classification_loss, iteration)
            tf.summary.scalar("validation.loss.%s.contr" % task, contrast_loss, iteration)

            tf.summary.scalar('validation.acc.%s.spenc' % task, speaker_encoder_acc, iteration)
            tf.summary.scalar('validation.acc.%s.spcla' % task, speaker_classification_acc, iteration)
            tf.summary.scalar('validation.acc.%s.texcl' % task, text_classification_acc, iteration)

            tf.summary.image("%s.mel_target" % task, plot_to_image(image_for_logger(y[0, :, :])), iteration)
            tf.summary.image("%s.mel_zpredicted" % task, plot_to_image(image_for_logger(y_pred.numpy()[0, :, :])),
                             iteration)
            self.writer.flush()
