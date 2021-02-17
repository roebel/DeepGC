"""
Compute mel-spectrograms mean and standard deviation after speaker encoder processing
"""
import os.path
import numpy as np
from fileio.iovar import load_var, save_var
from model import Parrot
from manage_model import build_model, restore_checkpoint, init_checkpoint_manager
from hparams import create_hparams, get_root_dir
from reader import TextMelIDLoader2, myDataLoader2
import tensorflow as tf
import sys


def estimate_mean_std(output_file, data_loader, model, num_sentences=np.inf):

    print('compute mel-spectrograms mean and standard deviation after speaker encoder processing')
    len_data_loader = len(data_loader)
    counter = 0
    for step in range(len_data_loader):
        if step % 50 == 0:
            print('%d/%d steps' % (step + 1, len_data_loader), end='\r')
        batch = data_loader[step]
        (text_input_phonelevel_padded, text_input_padded, mel_padded, mat_onehot_padded,
         expand_mat_padded, speaker_id, text_lengths, mel_lengths,
         stop_token_padded) = batch
        mel_reference = []  # unused
        _, embeddings = model([tf.transpose(mel_padded, (0, 2, 1)), mel_lengths], training=False)
        mel_input = embeddings.numpy().T
        if counter == 0:
            mel_frames = mel_input.shape[1]
            mel_sum = np.sum(mel_input, 1)
            mel_sum2 = np.sum(mel_input**2, 1)
        elif counter == num_sentences:
            break
        else:
            mel_frames += mel_input.shape[1]
            mel_sum += np.sum(mel_input, 1)
            mel_sum2 += np.sum(mel_input**2, 1)
        counter += batch_size
    print('number of sentences: %d' % mel_frames)
    print(counter)
    mel_mean = mel_sum.reshape((-1, 1))/mel_frames
    mel_var = mel_sum2.reshape((-1, 1))/mel_frames-mel_mean**2
    mel_std = np.sqrt(mel_var)

    data_dict = {'mean_mell': mel_mean, 'std_mell': mel_std}
    # make dir if necessary
    os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)
    # save dictionary
    save_var(output_file, data_dict)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="compute mel spectrograms mean and standard deviation after speaker encoding")
    parser.add_argument('--num', type=int, default=np.inf,
                        required=False, help='number of sentences used to compute mean/std')
    parser.add_argument('--model', type=str, required=True, help='directory with full VC model input')
    parser.add_argument('--ckpt', action="store_true", help="use checkpoint manager to load models")
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
    VCmodel = Parrot(hparams)
    build_model(VCmodel, hparams)
    if args.ckpt:
        restore_checkpoint(VCmodel, 1, checkpoint_path=os.path.join(args.model, "ckpt_store"), EPOCH='')
    else:
        VCmodel.load_weights(args.model)
    VCspkenc = VCmodel.speaker_encoder.speaker_encoder
    trainset = TextMelIDLoader2(hparams.root_dir, hparams.mel_training_list_filtered,
                                hparams.mel_mean_std, hparams.phone_training_list_filtered)
    batch_size = hparams.batch_size
    print('batch_size = %d' % batch_size)
    train_loader = myDataLoader2(trainset, batch_size=batch_size)
    output_file = os.path.join(hparams.database_root_dir, hparams.mel_mean_std_speaker_encoder)
    print(output_file)
    estimate_mean_std(output_file, train_loader, model=VCspkenc, num_sentences=args.num)
