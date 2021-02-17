"""
Compute mean and std of mel spectrum for file list.
"""
import os.path
import numpy as np
from fileio.iovar import load_var, save_var
import random
import sys
sys.path.append(('..'))
from hparams import create_hparams, get_root_dir


def estimate_mean_std(root, mel_list_file, output_mean_std_mel, num_sentences=np.inf):
    """
    use the training data for estimating mean and standard deviation.
    use the full data list !!!
    compute the sum and the sum of squares of the data without loading one data file at a time
    Parameters
    ----------
    root: str
        root dir of database
    mel_list_file: str
        mel file list (relative path)
    output_mean_std_mel: str
        output filename (relative path)
    num_sentences: int
        number of sentences to estimate the mean and std
    Returns
    -------
        nothing, data saved in os.path.join(root, output_mean_std_mel)
    """
    print('compute mel-spectrograms mean and standard deviation')
    # read list of mel-spec files
    with open(os.path.join(root, mel_list_file)) as f:
        lines = f.readlines()
    mel_file_list = [os.path.join(root, line.strip()) for line in lines]
    # shuffle list
    random.shuffle(mel_file_list)

    for counter, mel_filename in enumerate(mel_file_list):
        mel_input = load_var(mel_filename)['mell']
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
    print('number of frames: %d' % mel_frames)
    mel_mean = mel_sum.reshape((-1, 1))/mel_frames
    mel_var = mel_sum2.reshape((-1, 1))/mel_frames-mel_mean**2
    mel_std = np.sqrt(mel_var)

    data_dict = {'mean_mell': mel_mean, 'std_mell': mel_std}
    out_file = os.path.join(root, output_mean_std_mel)
    # make dir if necessary
    os.makedirs(os.path.abspath(os.path.dirname(out_file)), exist_ok=True)
    # save dictionary
    save_var(out_file, data_dict)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="compute mel spectrograms mean and standard deviation")
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs', default=None)
    parser.add_argument('--num', '-n', type=int, default=np.inf,
                        required=False, help='number of sentences used to compute mean/std')
    parser.add_argument('--db_root_dir', type=str, default=None,
                        help='root dir containing the db_config.py file to be used for training, if not given this will be determined from '
                             'roots.yml located in the same directry as the present script  (Def: read from db_roots.yml)',
                        )

    args = parser.parse_args()
    # Configuration ###########
    db_root_dir = get_root_dir(config_file=os.path.join(os.path.dirname(__file__), "db_roots.yaml"),
                               dir_tag="root_dir", default_dir=args.db_root_dir)

    hparams = create_hparams(args.hparams, root_dir=db_root_dir)
    estimate_mean_std(hparams.database_root_dir, hparams.mel_training_list_filtered,
                      hparams.mel_mean_std, num_sentences=args.num)
