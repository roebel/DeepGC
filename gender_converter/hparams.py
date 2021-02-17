# import tensorflow as tf

import yaml
import platform
import getpass
from hparammodule.hparam import HParams
# from text import symbols


def get_root_dir(config_file, dir_tag="root_dir", default_dir = None):
    """
    Read db_root entries for current user and current machine
    Args:
        config_file ():
        dir_tag ():
        default_dir ():

    Returns:

    """
    if default_dir :
        return default_dir
    else:
        with open(config_file , "r") as fi:
            db_roots_dict = yaml.safe_load(fi)

        user = getpass.getuser()
        if user not in db_roots_dict:
            raise RuntimeError(f"train.py::error::cannot find user {user} in db_root.yaml!"
                               f" either add your db root dirs into db_roots.yaml or specify root dir on the command line ")
        user_roots = db_roots_dict[user]
        hostname = platform.node().split(".")[0]
        # handle jeanzay where we may have different suffixes
        for hh in user_roots:
            if hostname.startswith(hh):
                hostname = hh

        if hostname not in user_roots:
            raise RuntimeError(f"train.py::error::cannot find host {hostname} for user {user} in db_roots.yaml, "
                               f"either add your db root dirs into db_roots.py or specify root dir on the command line ")

        rootdir_config = user_roots[hostname]

        if dir_tag not in rootdir_config:
            raise RuntimeError(f"train.py::error::cannot find dir info for {dir_tag} in {config_file}!"
                               f" either add your db root dirs into db_roots.yaml or specify root dir on the command line ")
        rootdir = rootdir_config[dir_tag]

        return rootdir


def create_hparams(hparams_string=None, root_dir=None, cmu_root_dir=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=200,
        iters_per_checkpoint=1000,
        max_chkpt_to_keep=10,  # number of checkpoints to keep
        seed=1234,
        distributed_run=False,
        cudnn_enabled=True,
        cudnn_benchmark=False,

        ################################
        # Data Parameters              #
        ################################
        database_root_dir=root_dir,
        root_dir=root_dir,
        cmu_root_dir=cmu_root_dir,

        mel_training_list='list_108_i/mel_train.list',
        mel_validation_list='list_108_i/mel_eval.list',
        phone_training_list='list_108_i/phone_train.list',
        phone_validation_list='list_108_i/phone_eval.list',

        mel_training_list_filtered='list_108_filtered_i/mel_train.list',
        phone_training_list_filtered='list_108_filtered_i/phone_train.list',

        mel_mean_std='normalize_i/mel_mean_std.npy',

        ################################
        # Data Parameters              #
        ################################
        n_mel_channels=80,
        n_symbols=42,
        n_speakers=108,
        perc_train_sentences=0.9,
        predict_spectrogram=False,

        fine_tune=False,
        n_speakers_finetune=2,
        speaker_A='bdl',  # males
        speaker_B='rms',
        ################################
        # Model Parameters             #
        ################################
        max_time_steps=600,

        symbols_embedding_dim=512,

        # Text Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        text_encoder_dropout=0.5,
        use_text_encoder_dilatation=False,

        # mel Encoder parameters
        mel_encoder_hidden_dim=512,
        mel_embedding_dim=512,
        mel_encoder_dropout=0.2,  # 0.2,
        use_mel_encoder_dilatation=False,

        # Audio Encoder parameters
        spemb_input=False,
        n_frames_per_step_encoder=1,
        audio_encoder_hidden_dim=512,
        AE_attention_dim=128,
        AE_attention_location_n_filters=32,
        AE_attention_location_kernel_size=51,
        beam_width=10,

        # hidden activation 
        # relu linear tanh
        hidden_activation='tanh',

        # Speaker Encoder parameters
        speaker_encoder_hidden_dim=256,
        speaker_encoder_dropout=0.2,
        speaker_embedding_dim=128,


        # Speaker Classifier parameters
        SC_hidden_dim=512,
        SC_n_convolutions=3,
        SC_kernel_size=1,

        # Decoder parameters
        decoder_hidden_dim=128,

        feed_back_last=True,
        n_frames_per_step_decoder=1,
        decoder_rnn_dim=512,
        decoder_prenet=False,
        prenet_dim=[256, 256],
        max_decoder_steps=1000,
        stop_threshold=0.5,
    
        # Attention parameters
        attention_rnn_dim=512,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=17,

        # PostNet parameters
        postnet_n_convolutions=5,
        postnet_dim=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,

        # L1 reconstruction cost
        use_subband_recons=False,

        # speaker classification hidden mel rate
        spksclassif_at_mel_rate=False,
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        learning_rate_factor=0.95,
        weight_decay=1e-6,
        grad_clip_thresh=5.0,
        batch_size=32,

        use_contrastive_loss=True,
        normalize_loss_hidden=True,
        contrastive_loss_w=30.0,
        speaker_adversial_loss_type='l2',

        speaker_encoder_loss_w=1.0,
        text_classifier_loss_w=1.0,
        speaker_adversial_loss_w=20.,
        speaker_classifier_loss_w=0.1,
        ce_loss=False,

        gender_adversarial_loss_w=1.,
        gender_use_adversarial_discriminator=True,  # without fader loss if False
        deeper_gender_discriminator_network=False,
        gender_autoencoder_loss_w=1.,
        gender_latent_dim=60,
        gender_autoencoder_error_type='mae',  # mse or dotproduct or mae
        gender_pretrain=False,  # do not change svp!
        gender_pretrain_epochs=3,
        gender_decoder_trainable_gain=False,

        # perturbe the pre-train classifier
        # solution 1
        sexprediscriminator_addnoise=False,  # add noise to speaker embedding to train "dummy" discriminator
        noisescale=1.,

        mel_mean_std_speaker_encoder='standardize_i/mel_mean_std_speaker_encoder.npy',
        speaker_encoder_data_standardize=True,

        soft_gender_id=False
    )

    if hparams_string:
        # tf.logging.info('Parsing command line hparams: %s', hparams_string)
        print('Parsing command line hparams: %s' % hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        # tf.logging.info('Final parsed hparams: %s', list(hparams.values()))
        print('Final parsed hparams: %s')
        print(list(hparams.values()))
        pass
    return hparams
