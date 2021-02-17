import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from model import Parrot
from debugprint import print_debug
import sys
import os


def create_model(hparams):
    # define the model's network
    print('Instantiate Model')
    model = Parrot(hparams)
    return model


def build_model(model, hparams, use_text=True, training=True):

    print("Build model")
    model.build([(None, hparams.n_mel_channels, None), (None, None,), (None, 1,),
                 (None, None, None), (None, None, hparams.n_symbols),
                 (None, 1,), [(None, hparams.n_mel_channels, None), (None, 1,)]])
    model.summary()
    return model


def restore_checkpoint(model, max_chkpt_to_keep, checkpoint_path, optimizer_main=None, optimizer_spkclassif=None,
                       EPOCH=''):
    # restore model checkpoint
    print('Restore model checkpoint')
    if optimizer_main is None:
        checkpoint = tf.train.Checkpoint(model=model)
    else:
        checkpoint = tf.train.Checkpoint(model=model, sgd_main=optimizer_main, sgd_sc=optimizer_spkclassif)
    manager_checkpoint = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path,
                                                    max_to_keep=max_chkpt_to_keep)

    if EPOCH == '':
        if manager_checkpoint.latest_checkpoint:
            tf.print("-----------Restoring from {}-----------".format(manager_checkpoint.latest_checkpoint))
            val = checkpoint.restore(manager_checkpoint.latest_checkpoint)
            if optimizer_main is None:
                val.expect_partial()
            # EPOCH = manager_checkpoint.latest_checkpoint.split('/')[-1][5:]
        else:
            sys.exit('no checkpoint found')
    else:
        checkpoint_fname = os.path.join(checkpoint_path, 'ckpt-' + str(EPOCH))
        tf.print("-----------Restoring from {}-----------".format(checkpoint_fname))
        val = checkpoint.restore(checkpoint_fname)
        if optimizer_main is None:
            val.expect_partial()


def init_checkpoint_manager(model, optimizer_main, optimizer_spkclassif, output_directory,
                            max_chkpt_to_keep, warmup=False, restore=True):
    # ## Model Checkpoints : Initialise or Restore
    checkpoint = tf.train.Checkpoint(sgd_main=optimizer_main, sgd_sc=optimizer_spkclassif, model=model)
    manager_checkpoint = tf.train.CheckpointManager(checkpoint,
                                                    directory=os.path.join(output_directory, "ckpt_store"),
                                                    max_to_keep=max_chkpt_to_keep)
    # restore model checkpoint
    if restore:
        if warmup:
            print('WARMUP init/restore model checkpoint')
            if optimizer_main is None:
                sys.exit('you need to provide optimizers for warmup')

            if manager_checkpoint.latest_checkpoint:
                tf.print("-----------Restoring from {}-----------".format(manager_checkpoint.latest_checkpoint))
                # check if full checkpoint has been loaded
                val = checkpoint.restore(manager_checkpoint.latest_checkpoint).expect_partial()
                # assert does not work with Adam optimizer
                # val.assert_consumed()
            else:
                sys.exit('no checkpoint found')

        else:
            # normal init
            val = checkpoint.restore(manager_checkpoint.latest_checkpoint)
            if optimizer_main is None:
                val.expect_partial()

            if manager_checkpoint.latest_checkpoint:
                print("Restored from {}".format(manager_checkpoint.latest_checkpoint))
            else:
                print("Initializing from scratch.")
    else:
        print('DO NOT RESTORE MODEL IN MANAGER INITIALIZATION...')

    return manager_checkpoint


def export_checkpoint(model_type, hparams, checkpoint_dir, export_dir):

    # ## Instantiate model
    if model_type == "vc":
        model = Parrot(hparams)
    elif model_type == "genderdiscriminator":
        from modelgender import SexGParrot
        model = SexGParrot(hparams)
    elif model_type == "genderae":
        from modelgender import GParrot
        model = GParrot(hparams)
    else:
        import sys
        sys.exit('unknown model type')
    build_model(model, hparams)

    # ## Model Checkpoints : Restore
    print_debug(checkpoint_dir)
    restore_checkpoint(model, 10, checkpoint_dir)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    weights_file = os.path.join(export_dir, model_type + "_weights.tf")
    # save weights in .tf format
    model.save_weights(weights_file, save_format='tf')
