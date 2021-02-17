import tensorflow as tf
from tensorflow.keras.layers import Layer
from debugprint import print_debug
from .utils import get_mask_from_lengths


class ParrotLoss():
    def __init__(self, hparams):
        super(ParrotLoss, self).__init__()
        self.hidden_dim = hparams.encoder_embedding_dim
        self.mel_hidden_dim = hparams.mel_embedding_dim

        self.contr_w = hparams.contrastive_loss_w
        self.spenc_w = hparams.speaker_encoder_loss_w

        self.texcl_w = hparams.text_classifier_loss_w
        self.spadv_w = hparams.speaker_adversial_loss_w
        self.spcla_w = hparams.speaker_classifier_loss_w
        self.n_symbols = hparams.n_symbols

        self.fine_tune = hparams.fine_tune

        # speaker classif logit from mel hidden can be at text rate or at frame rate
        self.spksclassif_at_mel_rate = hparams.spksclassif_at_mel_rate

        if 'speaker_adversial_loss_type' in hparams:
            # adevrsarial loss measures (l2, l1, KL))
            self.speaker_adversial_loss_type = hparams.speaker_adversial_loss_type
        else:
            # default (from the original paper/code)
            self.speaker_adversial_loss_type = 'l2'
        print_debug('spk adv loss type: ' + self.speaker_adversial_loss_type)

    def contrastive_loss(self, text_hidden, mel_hidden, mel_lengths, eps=1e-5):
        """
        Zhang's basic constrastive loss
        """
        # ### CONTRASTIVE LOSS
        n_frames = mel_hidden.shape[1]  # n_frames = T
        # 1) contrastive mask #
        # # [B, T] -> [B, T, T] (tile)
        contrast_mask1 = tf.tile(tf.expand_dims(get_mask_from_lengths(mel_lengths), axis=2), [1, 1, n_frames])
        # # [B, T] -> [B, T, T] (tile)
        contrast_mask2 = tf.tile(tf.expand_dims(get_mask_from_lengths(mel_lengths), axis=1), [1, n_frames, 1])
        # # [B, T, T]
        contrast_mask = tf.cast(contrast_mask1 & contrast_mask2, tf.float32)

        # text_hidden [B, T, emb_size]
        # mel_hidden [B, T, emb_size]
        text_hidden_normed = text_hidden / (tf.norm(text_hidden, axis=2, keepdims=True) + eps)
        mel_hidden_normed = mel_hidden / (tf.norm(mel_hidden, axis=2, keepdims=True) + eps)

        # (x - y) ** 2 = x ** 2 + y ** 2 - 2xy
        # [batch_size, T, 1]
        distance_matrix_xx = tf.reduce_sum(text_hidden_normed ** 2, axis=2, keepdims=True)
        distance_matrix_yy = tf.reduce_sum(mel_hidden_normed ** 2, axis=2)
        # [batch_size, 1, T]
        distance_matrix_yy = tf.expand_dims(distance_matrix_yy, axis=1)
        # [batch_size, T, T]
        distance_matrix_xy = text_hidden_normed @ tf.transpose(mel_hidden_normed, (0, 2, 1))
        # [batch_size, T, T]
        distance_matrix = distance_matrix_xx + distance_matrix_yy - 2 * distance_matrix_xy

        identity_mat = tf.eye(distance_matrix.shape[1])
        margin = 1.

        contrast_loss = identity_mat * distance_matrix + \
                        (1. - identity_mat) * tf.maximum(margin - distance_matrix, tf.zeros_like(distance_matrix))
        contrast_loss = tf.reduce_sum(contrast_loss*contrast_mask) / tf.reduce_sum(contrast_mask)
        return contrast_loss

    def compute_loss(self, model_outputs, targets, speaker_target, input_text=False, eps=1e-5):

        (predicted_mel, predicted_mel_post, mel_lengths, text_lengths,
         speaker_logit_from_mel, speaker_logit_from_mel_hidden_text_or_mel_rate,
         expand_mat_padded, text_input_padded, text_hidden, mel_hidden, mel_hidden_text_or_mel_rate,
         text_logit_from_mel_hidden, text_target_text_level, mat_onehot_padded) = model_outputs
        mel_target = targets

        mel_mask = get_mask_from_lengths(mel_lengths)
        mel_mask = tf.expand_dims(mel_mask, axis=1)
        # mel_mask = tf.keras.backend.cast(tf.tile(mel_mask, [1, mel_target.shape[1], 1]), dtype='float32')

        # replicate mel_mask over mel features axis
        mel_mask = tf.tile(tf.keras.backend.cast(mel_mask, dtype='float32'), [1, mel_target.shape[1], 1])
        # n_frames = mel_hidden.shape[1]  # n_frames = T
        recon_loss = tf.reduce_sum(tf.abs(mel_target-predicted_mel)*mel_mask)/tf.reduce_sum(mel_mask)
        recon_post_loss = tf.reduce_sum(tf.abs(mel_target-predicted_mel_post)*mel_mask)/tf.reduce_sum(mel_mask)

        # contrastive loss
        contrast_loss = self.contrastive_loss(text_hidden, mel_hidden, mel_lengths, eps)
        if not self.fine_tune:
            # speaker classification loss from mel speaker space, at text frame rate
            # speaker_logit_from_mel_int = tf.cast(speaker_logit_from_mel, tf.int16)
            speaker_encoder_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)\
                (speaker_target, speaker_logit_from_mel)
            predicted_speaker = tf.cast(tf.math.argmax(speaker_logit_from_mel, axis=1), dtype=tf.int16)
            speaker_encoder_acc = tf.reduce_sum(tf.cast((predicted_speaker == speaker_target), tf.float32)) \
                                  / speaker_target.shape[0]
        else:
            speaker_encoder_loss = tf.convert_to_tensor(0., dtype=tf.dtypes.float32)
            speaker_encoder_acc = tf.convert_to_tensor(0., dtype=tf.dtypes.float32)

        if self.fine_tune:
            n_speakers = 2
        else:
            n_speakers = speaker_logit_from_mel_hidden_text_or_mel_rate.shape[2]
        n_text_frames = speaker_logit_from_mel_hidden_text_or_mel_rate.shape[1]

        text_mask = get_mask_from_lengths(text_lengths)
        sc_mel_mask = get_mask_from_lengths(mel_lengths)  # mask for speaker classifier at mel rate
        if not self.fine_tune:
            text_mask = tf.expand_dims(text_mask, axis=1)
            sc_mel_mask = tf.expand_dims(sc_mel_mask, axis=1)
        text_mask_float = tf.keras.backend.cast(text_mask, dtype='float32')
        sc_mel_mask_float = tf.keras.backend.cast(sc_mel_mask, dtype='float32')
        # # speaker classification losses
        # # fader losses
        # speaker classification loss from mel linguistic space
        if self.spksclassif_at_mel_rate:
            sc_mask_float = sc_mel_mask_float
        else:
            sc_mask_float = text_mask_float

        if self.fine_tune:
            # there is only 1 dimension for the speaker "code" (2 speakers!)
            # these two lines change
            speaker_logit_flatten = tf.keras.backend.flatten(speaker_logit_from_mel_hidden_text_or_mel_rate)
            predicted_speaker = tf.cast(speaker_logit_flatten > 0., dtype=tf.int16)

            speaker_target_ling = tf.tile(tf.expand_dims(speaker_target, axis=1), [1, n_text_frames])
            speaker_target_flatten = tf.keras.backend.flatten(speaker_target_ling)

            sc_mask_float = tf.keras.backend.flatten(sc_mask_float)
            speaker_classification_acc = tf.reduce_sum(tf.cast((predicted_speaker == speaker_target_flatten),
                                                               tf.float32) * sc_mask_float) \
                                         / tf.reduce_sum(sc_mask_float)
            # this line changes
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True) \
                (speaker_target_flatten, speaker_logit_flatten)
            speaker_classification_loss = tf.reduce_sum(loss * sc_mask_float) / tf.reduce_sum(sc_mask_float)

            # speaker adversival loss from mel hidden at frame rate
            if self.speaker_adversial_loss_type == 'l2':
                loss = tf.math.pow(tf.abs(tf.nn.sigmoid(speaker_logit_flatten) - 0.5), 2)
            elif self.speaker_adversial_loss_type == 'l1':
                loss = tf.abs(tf.nn.sigmoid(speaker_logit_flatten) - 0.5)
            elif self.speaker_adversial_loss_type == 'KL':
                # use inverse Kullback-Leibler divergence for 2 speakers = 2 probabilities p and 1-p
                epsilon = 1e-12  # to avoid problems with log
                ref_prob = 1. / n_speakers
                target_prob = (1 - epsilon) * tf.nn.sigmoid(speaker_logit_flatten) + epsilon
                loss = (1-target_prob)*tf.math.log((1-target_prob)/ref_prob) + target_prob*tf.math.log(target_prob/ref_prob)

            speaker_adversial_loss = tf.reduce_sum(loss * sc_mask_float) / tf.reduce_sum(sc_mask_float)

        else:
            speaker_logit_flatten = tf.reshape(speaker_logit_from_mel_hidden_text_or_mel_rate, [-1, n_speakers])
            predicted_speaker = tf.cast(tf.math.argmax(speaker_logit_flatten, axis=1), dtype=tf.int16)

            speaker_target_ling = tf.tile(tf.expand_dims(speaker_target, axis=1), [1, n_text_frames])
            speaker_target_flatten = tf.keras.backend.flatten(speaker_target_ling)

            speaker_classification_acc = tf.reduce_sum(tf.cast((predicted_speaker == speaker_target_flatten),
                                                  tf.float32)*tf.keras.backend.flatten(sc_mask_float))\
                                         / tf.reduce_sum(sc_mask_float)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                                 from_logits=True)\
                (speaker_target_flatten, speaker_logit_flatten)
            speaker_classification_loss = tf.reduce_sum(loss*tf.keras.backend.flatten(sc_mask_float))\
                                        / tf.reduce_sum(sc_mask_float)

            # speaker adversarial loss from mel hidden at frame rate
            flatten_target = 1. / n_speakers   # * tf.ones_like(speaker_logit_flatten)

            if self.speaker_adversial_loss_type == 'l2':
                loss = tf.math.pow(tf.abs(tf.nn.softmax(speaker_logit_flatten, axis=1) - flatten_target), 2)
            elif self.speaker_adversial_loss_type == 'l1':
                loss = tf.abs(tf.nn.softmax(speaker_logit_flatten, axis=1) - flatten_target)
            elif self.speaker_adversial_loss_type == 'KL':
                # use inverse Kullback-Leibler divergence
                epsilon = 1e-12  # to avoid problems with log
                ref_prob = 1. / n_speakers  # flatten_target
                target_prob = (1 - epsilon) * tf.nn.softmax(speaker_logit_flatten, axis=1) + epsilon
                loss = target_prob*tf.math.log(target_prob/ref_prob)

            # not sure of this (mask)
            mask = tf.reshape(tf.tile(tf.transpose(sc_mask_float, (0, 2, 1)),
                                      [1, 1, n_speakers]), [-1, n_speakers])

            speaker_adversial_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

        # text classification loss
        # text classification loss from mel hidden at text rate
        # compress from mel rate to text rate (normalize by the phone durations)
        text_logit_from_mel_hidden_text_rate = expand_mat_padded @ text_logit_from_mel_hidden

        # input the actual text at phone level rather than compress from mel level!
        text_logit_flatten = tf.reshape(text_logit_from_mel_hidden_text_rate, [-1, self.n_symbols])
        text_target_flatten = tf.keras.backend.flatten(text_target_text_level)

        predicted_text = tf.cast(tf.math.argmax(text_logit_flatten, axis=1), dtype=tf.int16)

        text_classification_acc = tf.reduce_sum(tf.cast((predicted_text == text_target_flatten),
                                                tf.float32)*tf.keras.backend.flatten(text_mask_float))\
                                     / tf.reduce_sum(text_mask_float)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                             from_logits=True)\
            (text_target_flatten, text_logit_flatten)
        text_classification_loss = tf.reduce_sum(loss*tf.keras.backend.flatten(text_mask_float)) / \
                                   tf.reduce_sum(text_mask_float)

        loss_list = [recon_loss, recon_post_loss, speaker_encoder_loss, speaker_classification_loss,
                     speaker_adversial_loss, text_classification_loss, contrast_loss]
        accuracy_list = [speaker_encoder_acc, speaker_classification_acc, text_classification_acc]

        combined_loss1 = recon_loss + self.spenc_w * speaker_encoder_loss + self.spadv_w * speaker_adversial_loss + \
                         self.texcl_w * text_classification_loss + self.contr_w * contrast_loss + recon_post_loss
        # self.contr_w * contrast_loss + \
        #                  + self.texcl_w * text_classification_loss + \
        #                 self.spadv_w * speaker_adversial_loss

        combined_loss2 = self.spcla_w * speaker_classification_loss

        return loss_list, accuracy_list, combined_loss1, combined_loss2
