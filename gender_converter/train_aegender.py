#! /usr/bin/env python3

import os
import time
import argparse
import numpy as np
import sys
from debugprint import print_debug  
# from model import Parrot, ParrotLoss, lcm
from reader import TextMelIDLoader2, myDataLoader2
from hparams import create_hparams, get_root_dir
from model import Parrot
from modelgender import GParrot, SexGParrot
from modelgender.lossgender import GParrotLoss
from logger_aegender import GParrotLogger
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
    # Get data, data loaders ready

    trainset = TextMelIDLoader2(hparams.root_dir, hparams.mel_training_list_filtered,
                                hparams.mel_mean_std, hparams.phone_training_list_filtered)
    valset = TextMelIDLoader2(hparams.root_dir, hparams.mel_validation_list,
                              hparams.mel_mean_std, hparams.phone_validation_list)
    collate_fn = []

    train_loader = myDataLoader2(trainset, batch_size=hparams.batch_size)

    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
    logger = GParrotLogger(os.path.join(output_directory, log_directory))

    return logger


def validate(model, criterion, valset, iteration, logger, batch_size, valstep=100):
    '''Handles all the validation scoring and printing'''
    print('')
    print("validate model")

    val_loader = myDataLoader2(valset, batch_size=batch_size)
    # val_loader.randomize()
    val_loss = 0.0
    # number of losses and accuracies
    n_losses = 5
    n_acces = 2
    reduced_val_losses = np.zeros([n_losses], dtype=np.float32)
    reduced_val_acces = np.zeros([n_acces], dtype=np.float32)

    len_val_loader = min(len(val_loader), valstep)
    for step in range(0, len_val_loader):
        # start = time.time()

        use_text = step % 2 == 0
        if step % 50 == 0:
            print('%d/%d steps' % (step, min(len(val_loader), valstep)), end='\r')
        (text_input_phonelevel_padded, text_input_padded, mel_padded, mat_onehot_padded,
         expand_mat_padded, speaker_id, text_lengths, mel_lengths,
         stop_token_padded) = val_loader[step]

        mel_reference = []  # unused
        (speaker_logit_from_mel, speaker_embedding, speaker_embedding_out, gender_logit_from_gdisc) = \
            model([mel_padded, text_input_padded, mel_lengths, expand_mat_padded, mat_onehot_padded,
                   speaker_id, mel_reference], use_text, training=False, do_voice_conversion=False)

        # Compute the loss value for this minibatch.
        model_outputs = [speaker_logit_from_mel, speaker_embedding, speaker_embedding_out, gender_logit_from_gdisc]

        # targets = mel_padded
        gender_id = mat_onehot_padded  # from dirty hack
        loss_list, accuracy_list, combined_loss1, combined_loss2 = \
            criterion.compute_loss(model_outputs, gender_id, speaker_id)

        val_loss += combined_loss1 + combined_loss2
        reduced_val_losses += np.array([val.numpy() for val in loss_list])
        reduced_val_acces += np.array([acc.numpy() for acc in accuracy_list])

    val_loss /= len_val_loader
    reduced_val_acces /= len_val_loader
    reduced_val_losses /= len_val_loader

    print(("Validation loss {}: {:.5f}".format(iteration, float(val_loss))))
    logger.log_validation(val_loss, reduced_val_losses, reduced_val_acces, iteration)
    print('Done!')


def train_step(model, criterion, tvars_main, sgd_main, tvars_sc, sgd_sc,
               batch, use_text):
    (text_input_phonelevel_padded, text_input_padded, mel_padded, gender_id,
     expand_mat_padded, speaker_id, text_lengths, mel_lengths,
     stop_token_padded) = batch
    use_adversarial = hparams.gender_use_adversarial_discriminator
    with tf.GradientTape(persistent=use_adversarial) as tape:
        # with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        mel_reference = []  # unused
        (speaker_logit_from_mel, speaker_embedding,
         speaker_embedding_out, gender_logit_from_gdisc) = model([mel_padded, text_input_padded, mel_lengths,
                                                                 expand_mat_padded, gender_id,
                                                                 speaker_id, mel_reference],
                                                                use_text, training=True, do_voice_conversion=False)

        # Compute the loss value for this minibatch.
        model_outputs = [speaker_logit_from_mel, speaker_embedding, speaker_embedding_out, gender_logit_from_gdisc]
        # targets = mel_padded
        loss_list, accuracy_list, combined_loss1, combined_loss2 = \
            criterion.compute_loss(model_outputs, gender_id, speaker_id)
    # apply gradient to all losses except speaker classifier loss
    grads = tape.gradient(combined_loss1, tvars_main)
    # gradient clipping
    # https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
    # see: https: // stackoverflow.com / questions / 36498127 / how - to - apply - gradient - clipping - in -tensorflow
    grads, grad_norm_main = tf.clip_by_global_norm(grads, hparams.grad_clip_thresh)
    sgd_main.apply_gradients(zip(grads, tvars_main))

    # apply gradient to speaker classifier loss
    if use_adversarial:
        grads = tape.gradient(combined_loss2, tvars_sc)
        sgd_sc.apply_gradients(zip(grads, tvars_sc))

    return loss_list, accuracy_list, combined_loss1, combined_loss2, grad_norm_main


def train_batch(model, criterion, tvars_main, sgd_main, tvars_sc, sgd_sc, batch,
                use_text, iteration, logger, output_directory, val_set, learning_rate, manager_checkpoint):
    valstep = 500  # maximum number of batches for validation (all of them with batch_size = 32)
    start = time.time()

    (loss_list, accuracy_list, combined_loss1,
     combined_loss2, grad_norm_main) = train_step(model, criterion, tvars_main,
                                                  sgd_main, tvars_sc, sgd_sc, batch,
                                                  use_text)
    total_loss = combined_loss1 + combined_loss2
    duration = time.time() - start

    if iteration % 100 == 0:
        print("Train {} {:.6f}\tGrad Norm {:.5f}\t{:.2f}\t{:.2f}s/it".format(iteration, float(total_loss),
                                                                             grad_norm_main,
                                                                             model.gdecoder.gainaftertanh.numpy(),
                                                                             duration), end='\n')

    if iteration > 0:
        logger.log_training(total_loss, loss_list, accuracy_list, grad_norm_main, learning_rate,
                            duration, iteration)

    if (iteration % hparams.iters_per_checkpoint == 0):
        save_path = manager_checkpoint.save(checkpoint_number=iteration)
        print_debug(save_path)
        validate(model, criterion, val_set, iteration, logger, hparams.batch_size,
                 valstep=valstep)


def train(output_directory, log_directory, hparams, warmup=False, VCmodelspkenc_weights=None,
          sexdiscriminator_weights=None):

    train_loader, val_set, _, = prepare_dataloaders(hparams)
    # we call train_batch with train_loader[step] (tts) and train_loader[step+1] (vc),
    # so len_train_loader must be even
    if len(train_loader) % 2 == 1:
        len_train_loader = len(train_loader) - 1
    else:
        len_train_loader = len(train_loader)

    # learning_rate = hparams.learning_rate
    epochs = hparams.epochs

    # create and build model
    model = GParrot(hparams)
    build_model(model, hparams)
    # copy speaker encoder weights
    model.speaker_encoder.set_weights(VCmodelspkenc_weights)
    # copy pre-trained gender dicriminator weights
    model.gprediscriminator.set_weights(sexdiscriminator_weights)
    criterion = GParrotLoss(hparams)

    # learning_rate = 1e-4
    learning_rate = hparams.learning_rate
    sgd_main = tf.keras.optimizers.SGD(learning_rate=tf.Variable(learning_rate), momentum=0.9)
    sgd_sc = tf.keras.optimizers.SGD(learning_rate=tf.Variable(learning_rate), momentum=0.9)

    # sgd_main = tf.keras.optimizers.Adam(lr=tf.Variable(learning_rate))
    # sgd_sc = tf.keras.optimizers.Adam(lr=tf.Variable(learning_rate))
    print('learning rate = %e' % learning_rate)
    # trainable weight sets
    # split parameters into speaker classifier and the rest
    tvars_sc = model.sc_trainable_weights
    tvars_main = model.main_trainable_weights

    print_debug('number of main vars: %d' % len(tvars_main))
    print_debug('number of sc vars: %d' % len(tvars_sc))

    # ## Model Checkpoints : Initialise or Restore
    manager_checkpoint = init_checkpoint_manager(model, sgd_main, sgd_sc, output_directory,
                                                 hparams.max_chkpt_to_keep, warmup)
    if warmup:
        # full restore of latest checkpoint
        learning_rate = sgd_main.lr.numpy()
        iteration = sgd_main.iterations.numpy()
        min_epoch = iteration // len(train_loader)
        print_debug(min_epoch)
    else:
        iteration = 0
        min_epoch = 0

    print_debug('eargerly? ')
    print_debug(tf.executing_eagerly())

    logger = prepare_directories_and_logger(output_directory, log_directory)

    for epoch in range(min_epoch, min_epoch+epochs):
        print("\nStart of epoch %d" % (epoch,))

        epoch_tic = time.time()
        # Iterate over the batches of the dataset.
        # shuffle training sentences
        train_loader.randomize()
        for step in range(0, len_train_loader, 2):
            # ###
            #  VC
            # ###
            batchvc = train_loader[step+1]
            train_batch(model, criterion, tvars_main, sgd_main, tvars_sc, sgd_sc, batchvc,
                        False, iteration, logger, output_directory, val_set, learning_rate, manager_checkpoint)
            iteration += 1
            # ###
            # TTS
            # ###
            batchtts = train_loader[step]
            train_batch(model, criterion, tvars_main, sgd_main, tvars_sc, sgd_sc, batchtts,
                        True, iteration, logger, output_directory, val_set, learning_rate, manager_checkpoint)
            iteration += 1

        print('')
        print('epoch %d, elapsed time %s' % (epoch, format_time(time.time() - epoch_tic)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default='logdir',
                        help='directory to save tensorboard logs (Def: %(default)s)')
    parser.add_argument('-igender', '--input_directory_gender', type=str, required=True,
                        help='directory with the gender discrimintor model input checkpoint (only one, latest)')
    parser.add_argument('-ivc', '--input_directory_vc', type=str, required=True,
                        help='directory with the VC model input checkpoint (only one, latest)')
    parser.add_argument('--cpu', action="store_true", help="do not lock a GPU, and run on CPU")
    parser.add_argument('--hparams', type=str, default=None, help='comma separated name=value pairs')
    parser.add_argument('--db_root_dir', type=str, default=None,
                        help='root dir containing the db_config.py file to be used for training, if not given this will be determined from '
                             'roots.yml located in the same directry as the present script  (Def: read from db_roots.yml)',
                        )
    parser.add_argument('--warmup', action="store_true", help="use last checkpoint for warmup")
    parser.add_argument("--eager_mode", action="store_true",
                    help="disable tf.function and force running in eager mode (Def: %(default)s)")

    args = parser.parse_args()
    if args.eager_mode:
        tf.config.experimental_run_functions_eagerly(True)

    db_root_dir = get_root_dir(config_file=os.path.join(os.path.dirname(__file__), "db_roots.yaml"),
                               dir_tag="root_dir", default_dir=args.db_root_dir)

    hparams = create_hparams(args.hparams, root_dir=db_root_dir)

    if args.cpu:
        print("using CPU, dev mode?")
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    # create and build model and load checkpoint
    # the aim is to get the speaker encoder weights and the pre-trained gender classifier weights

    # load gender discriminator weights
    sexmodel = SexGParrot(hparams)
    build_model(sexmodel, hparams)
    restore_checkpoint(sexmodel, 1, checkpoint_path=os.path.join(args.input_directory_gender, "ckpt_store"), EPOCH='')
    sexdiscriminator_weights = sexmodel.gprediscriminator.get_weights()
    VCspkenc_weights0 = sexmodel.speaker_encoder.get_weights()

    # ATTENTION
    # THIS IS NOT USEFUL BUT IS HERE TO BE SURE EVERYTHING WORKS
    # SHOULD BE REMOVED BECAUSE THESE WEIGHTS ARE ALSO IN THE SexGParrot MODEL
    VCmodel = Parrot(hparams)
    build_model(VCmodel, hparams)
    restore_checkpoint(VCmodel, 1, checkpoint_path=os.path.join(args.input_directory_vc, "ckpt_store"), EPOCH='')
    VCspkenc_weights = VCmodel.speaker_encoder.speaker_encoder.get_weights()
    # print_debug(VCspkenc_weights[0])

    # sys.exit(0)
    train(args.output_directory, args.log_directory, hparams, warmup=args.warmup,
          VCmodelspkenc_weights=VCspkenc_weights, sexdiscriminator_weights=sexdiscriminator_weights)



