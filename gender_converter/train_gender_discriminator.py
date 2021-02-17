#! /usr/bin/env python3

import os
import time
import argparse
import sys
from debugprint import print_debug
from reader import TextMelIDLoader2, myDataLoader2
from hparams import create_hparams, get_root_dir
from model import Parrot
from modelgender import SexGParrot
from manage_model import build_model, restore_checkpoint, init_checkpoint_manager
# uses tensorflow 2.x
import tensorflow as tf


def format_time(time_sec):
    # print time in hour:minute:second
    time_sec = int(time_sec)
    t_hour = time_sec//3600
    # time_sec = time_sec-t_hour*3600
    t_minute = (time_sec % 3600)//60
    t_sec = time_sec % 60
    t_string = str(t_hour) + ':' + str(t_minute) + ':' + str(t_sec)
    return t_string


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    # use mel_training_list_filtered2 and phone_training_list_filtered3
    trainset = TextMelIDLoader2(hparams.root_dir, hparams.mel_training_list_filtered,
                                hparams.mel_mean_std, hparams.phone_training_list_filtered)
    valset = TextMelIDLoader2(hparams.root_dir, hparams.mel_validation_list,
                              hparams.mel_mean_std, hparams.phone_validation_list)
    collate_fn = []

    train_loader = myDataLoader2(trainset, batch_size=hparams.batch_size)

    return train_loader, valset, collate_fn


def compute_loss(gender_logit_from_gdisc, gender_target):

    # Compute the loss value and accuracy
    gender_target_flatten = tf.keras.backend.flatten(gender_target)
    gender_logit_from_gpredisc_flatten = tf.keras.backend.flatten(gender_logit_from_gdisc)
    bce_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True) \
        (gender_target_flatten, gender_logit_from_gpredisc_flatten)
    gender_pre_classification_loss = tf.reduce_mean(bce_loss)

    predicted_gender = tf.cast(gender_logit_from_gpredisc_flatten > 0., dtype=tf.int16)
    gender_classification_acc = tf.reduce_mean(tf.cast(predicted_gender == gender_target_flatten, dtype=tf.float32))

    return gender_pre_classification_loss, gender_classification_acc


def val_step(model, batch):
    """
    validation of pre-gender classifier, one step
    Parameters
    ----------
    model
    batch

    Returns
    -------

    """
    (text_input_phonelevel_padded, text_input_padded, mel_padded, gender_id,
     expand_mat_padded, speaker_id, text_lengths, mel_lengths,
     stop_token_padded) = batch

    use_text = False  # does change anything here
    training = False  # it is validation, not training
    mel_reference = []  # unused
    (speaker_logit_from_mel, _, _, gender_logit_from_gdisc) = \
        model([mel_padded, text_input_padded, mel_lengths, expand_mat_padded,
               gender_id, speaker_id, mel_reference], use_text=use_text, training=training)

    # Compute the loss value and accuracy
    gender_pre_classification_loss, gender_classification_acc = compute_loss(gender_logit_from_gdisc, gender_id)

    return gender_pre_classification_loss, gender_classification_acc


# @tf.function
def train_step(model, tvars_gprec, sgd_gprec, batch):
    """
    training of pre-gender classifier, one step
    Parameters
    ----------
    model
    tvars_gprec
    sgd_gprec
    batch

    Returns
    -------

    """
    (text_input_phonelevel_padded, text_input_padded, mel_padded, gender_id,
     expand_mat_padded, speaker_id, text_lengths, mel_lengths,
     stop_token_padded) = batch
    with tf.GradientTape() as tape:
        mel_reference = []  # unused
        use_text = False  # does change anything here
        training = True  # it is training stage
        (speaker_logit_from_mel, _, _, gender_logit_from_gdisc) = \
            model([mel_padded, text_input_padded, mel_lengths, expand_mat_padded,
                   gender_id, speaker_id, mel_reference], use_text=use_text, training=training)

        # Compute the loss value and accuracy
        gender_pre_classification_loss, gender_classification_acc = compute_loss(gender_logit_from_gdisc, gender_id)

    # apply gradient
    grads = tape.gradient(gender_pre_classification_loss, tvars_gprec)
    sgd_gprec.apply_gradients(zip(grads, tvars_gprec))

    return gender_pre_classification_loss, gender_classification_acc


def train(output_directory, hparams, VCmodelspkenc_weights):

    train_loader, val_set, _, = prepare_dataloaders(hparams)
    len_train_loader = len(train_loader)
    val_loader = myDataLoader2(val_set, batch_size=hparams.batch_size)
    len_val_loader = len(val_loader)

    epochs = hparams.gender_pretrain_epochs
    print('NUMBER OF EPOCHS: ' + str(epochs))

    # create and build model
    model = SexGParrot(hparams)
    build_model(model, hparams)
    model.speaker_encoder.set_weights(VCmodelspkenc_weights)

    # Model Checkpoint : Initialise without restoring!!!
    sgd_main = tf.keras.optimizers.SGD(learning_rate=tf.Variable(1e-4), momentum=0.9)
    sgd_sc = tf.keras.optimizers.SGD(learning_rate=tf.Variable(1e-4), momentum=0.9)
    # only create checkpoint manager
    manager_checkpoint = init_checkpoint_manager(model, sgd_main, sgd_sc, output_directory,
                                                 hparams.max_chkpt_to_keep, warmup=False, restore=False)

    print_debug('eargerly? ')
    print_debug(tf.executing_eagerly())

    # optimizer and get trainable weights
    sgd_gprec = tf.keras.optimizers.SGD(learning_rate=tf.Variable(1e-3), momentum=0.9)
    tvars_gprec = model.gprec_trainable_weights
    print_debug('number of pre-train vars: %d' % len(tvars_gprec))

    iteration = 0
    for epoch in range(epochs):
        print("\nEpoch %d" % (epoch,))

        epoch_tic = time.time()
        # Iterate over the batches of the dataset.
        # shuffle training sentences
        train_loader.randomize()
        for step in range(len_train_loader):
            batch = train_loader[step]
            gender_classification_loss, gender_classification_acc = train_step(model, tvars_gprec, sgd_gprec, batch)
            iteration += 1
            if (iteration % 100 == 0) or (iteration == 0):
                print('iter %d,' % iteration, end=' ')
                print('gender discr, loss %.3f, acc %.3f' % (gender_classification_loss, gender_classification_acc))

        # validation
        val_gender_classification_loss = 0.
        val_gender_classification_acc = 0.
        for step in range(len_val_loader):
            batch = val_loader[step]
            val_gender_classification_step_loss, val_gender_classification_step_acc = val_step(model, batch)
            val_gender_classification_loss += val_gender_classification_step_loss
            val_gender_classification_acc += val_gender_classification_step_acc
            if (step % 50 == 0) or (step == 0):
                print('iter %d/%d,' % (step, len_val_loader), end='\r')
        print('')
        print('gender validation, loss %.3f, acc %.3f' % (val_gender_classification_loss/len_val_loader,
                                                         val_gender_classification_acc/len_val_loader))
        print('')
        print('epoch %d, elapsed time %s' % (epoch, format_time(time.time() - epoch_tic)))
    save_path = manager_checkpoint.save(checkpoint_number=iteration)
    print_debug(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train the so called pre-trained gender discriminator")
    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='directory to save checkpoints')
    parser.add_argument('-i', '--input_directory', type=str, required=True,
                        help='directory with full VC model input checkpoint (latest)')
    parser.add_argument('--cpu', action="store_true", help="do not lock a GPU, and run on CPU")
    parser.add_argument('--hparams', type=str, default=None, help='comma separated name=value pairs')
    parser.add_argument('--db_root_dir', type=str, default=None,
                        help='root dir containing the db_config.py file to be used for training, if not given this will be determined from '
                             'roots.yml located in the same directry as the present script  (Def: read from db_roots.yml)',
                        )
    parser.add_argument("--eager_mode", action="store_true",
                        help="disable tf.function and force running in eager mode (Def: %(default)s)")

    args = parser.parse_args()
    if args.eager_mode:
        tf.config.experimental_run_functions_eagerly(True)

    db_root_dir = get_root_dir(config_file=os.path.join(os.path.dirname(__file__), "db_roots.yaml"),
                               dir_tag="root_dir", default_dir=args.db_root_dir)

    hparams = create_hparams(args.hparams, root_dir=db_root_dir)
    hparams.gender_pretrain = True

    if args.cpu:
        print("using CPU, dev mode?")
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    # create and build model and load checkpoint
    # the aim is to get the speaker encoder weights (trained on the pre-train database with the complete VC system)
    VCmodel = Parrot(hparams)
    build_model(VCmodel, hparams)
    restore_checkpoint(VCmodel, 1, checkpoint_path=os.path.join(args.input_directory, "ckpt_store"), EPOCH='')
    VCspkenc_weights = VCmodel.speaker_encoder.speaker_encoder.get_weights()
    train(args.output_directory, hparams, VCmodelspkenc_weights=VCspkenc_weights)
