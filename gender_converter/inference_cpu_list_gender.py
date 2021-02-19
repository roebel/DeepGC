#! /usr/bin/env python3

import matplotlib
import matplotlib.pylab as plt
import os
import os.path
# set no GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np

import librosa
from librosa.util import nnls
from pysndfile import sndio

from tensorflow.keras.layers import Input
import tensorflow as tf
from fileio.iovar import load_var, save_var
from reader import TextMelIDLoader2, myDataLoader2, get_experiment_speaker_info, get_experiment_phn_info
from hparams import create_hparams, get_root_dir
from model import Parrot, FullGenderParrot
from modelgender import GParrot
from debugprint import print_debug
from manage_model import create_model, build_model, restore_checkpoint
import time


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def recover_wav_griffinlim(mel, n_fft=2048, win_length=800, hop_length=200, n_mels=80, melstd_gain=1.):
    data_dict = load_var(os.path.join(hparams.root_dir, hparams.mel_mean_std))
    mell_mean = data_dict['mean_mell'].astype('float32')
    mell_std = data_dict['std_mell'].astype('float32')

    mel1 = np.exp(melstd_gain * mel * mell_std + 1.0 * mell_mean)

    filters = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels)
    spec = nnls(filters, mel1)

    y = librosa.feature.inverse.griffinlim(spec, n_iter=50, hop_length=hop_length, win_length=win_length,
                                           window='hann',
                                           center=True, dtype=np.float32, length=None, pad_mode='reflect',
                                           momentum=0.8, init='random', random_state=None)
    return y


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="inference with gender modification (on CPU)")
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs (Def: %(default)s)',
                        default=None)
    parser.add_argument('--vcmodel', type=str, required=True,
                        help='VC model file')
    parser.add_argument('--gendermodel', type=str, required=True,
                        help='gender model file')
    parser.add_argument('--ckpt', action="store_true", help="use checkpoint manager to load models")
    parser.add_argument('--out', type=str, required=True,
                        help='generated audio and data files output directory')
    parser.add_argument('--task', type=str, default='vc',
                        required=False, help='task (vc or tts) (Def: %(default)s)')
    parser.add_argument('--mellist', type=str,
                        required=True, help='file with list of mel files')
    parser.add_argument('--phonelist', type=str,
                        required=True, help='file with list of phone files')
    parser.add_argument('--db_root_dir', type=str, default=None,
                        help='root dir containing the db_config.py file to be used for training, if not given this will be determined from '
                             'roots.yml located in the same directry as the present script  (Def: read from db_roots.yml)',
                        )
    parser.add_argument('--wav', action="store_true", required=False, help='compute wav files with Griffin-Lim (should use Waveglow instead), (Def: %(default)s)')

    args = parser.parse_args()
    print("inference with gender modification (on CPU)")
    mel_list = args.mellist
    phone_list_list = args.phonelist

    # Configuration ###########
    db_root_dir = get_root_dir(config_file=os.path.join(os.path.dirname(__file__), "db_roots.yaml"),
                               dir_tag="root_dir", default_dir=args.db_root_dir)

    hparams = create_hparams(args.hparams, root_dir=db_root_dir)
    _, sp2id, id2sp = get_experiment_speaker_info(hparams.root_dir)
    _, ph2id, id2ph = get_experiment_phn_info()

    # TTS or VC task?
    if args.task == 'vc':
        input_text = False
    elif args.task == 'tts':
        input_text = True
    else:
        sys.exit('unknown task (choose vc or tts)')

    # create and build VC model and load weights or checkpoint
    vcmodel = Parrot(hparams)
    build_model(vcmodel, hparams)
    if args.ckpt:
        checkpoint_vc_path = args.vcmodel
        restore_checkpoint(vcmodel, 1, checkpoint_path=os.path.join(args.vcmodel, "ckpt_store"), EPOCH='')
    else:
        vcmodel.load_weights(args.vcmodel)

    # create and build gender model and load checkpoint or weights
    gendermodel = GParrot(hparams)
    build_model(gendermodel, hparams)
    if args.ckpt:
        checkpoint_gender_path = args.gendermodel
        restore_checkpoint(gendermodel, 1, checkpoint_path=os.path.join(args.gendermodel, "ckpt_store"), EPOCH='')
    else:
        gendermodel.load_weights(args.gendermodel)

    # create and build full model
    model = FullGenderParrot(hparams)
    build_model(model, hparams)

    # copy VC weights
    model.speaker_encoder.set_weights(vcmodel.speaker_encoder.get_weights())
    model.embedding.set_weights(vcmodel.embedding.get_weights())
    model.mel_encoder.set_weights(vcmodel.mel_encoder.get_weights())
    model.text_encoder.set_weights(vcmodel.text_encoder.get_weights())
    model.decoder.set_weights(vcmodel.decoder.get_weights())
    model.postnet.set_weights(vcmodel.postnet.get_weights())
    model.merge_net.set_weights(vcmodel.merge_net.get_weights())
    model.speaker_classifier.set_weights(vcmodel.speaker_classifier.get_weights())

    # copy gender weights
    model.gencoder.set_weights(gendermodel.gencoder.get_weights())
    model.gdecoder.set_weights(gendermodel.gdecoder.get_weights())
    model.gdicriminator.set_weights(gendermodel.gdicriminator.get_weights())
    model.gprediscriminator.set_weights(gendermodel.gprediscriminator.get_weights())

    # dataset
    test_set = TextMelIDLoader2(hparams.root_dir, mel_list,
                                hparams.mel_mean_std, phone_list_list,
                                mel_phone_lists_relative_to_root=False)
    num_sentences = test_set.__len__()
    sample_list = test_set.mel_file_list
    # collate_fn = []
    # one file per batch
    test_loader = myDataLoader2(test_set, batch_size=1)

    # inference mode, normalization data etc
    task = 'tts' if input_text else 'vc'
    path_save = task
    path_save += '_' + args.out
    os.makedirs(path_save, exist_ok=True)

    # load mell mean/std normalization data
    data_dict = load_var(os.path.join(hparams.root_dir, hparams.mel_mean_std))
    mell_mean = data_dict['mean_mell'].astype('float32')
    mell_std = data_dict['std_mell'].astype('float32')

    for gender_type in ['original', 'degender', 'estgender', 'invgender']:

        if gender_type == 'original' or gender_type == 'degender':
            use_true_gender_id = True  # use gender_id
        elif gender_type == 'estgender' or gender_type == 'invgender':
            use_true_gender_id = False  # use predicted gender_id
        else:
            import sys
            sys.exit('unknown type (original, estgender, degender, invgender)')
        sr = 16000
        # inference (no gradient computation)
        errs = 0
        totalphs = 0

        for step in range(len(test_loader)):
            i = step
            tic = time.time()
            # get batch
            batch = test_loader[step]
            (text_input_phonelevel_padded, text_input_padded, mel_padded, mat_onehot_padded,
             expand_mat_padded, speaker_id, text_lengths, mel_lengths,
             stop_token_padded) = batch
            gender_id = mat_onehot_padded  # dirty hack
            # reference speaker mel-spec (UNUSED here)
            reference_mel = mel_padded
            # process file names
            basedir, mel_filename = os.path.split(test_set.mel_file_list[step])
            sample_id = mel_filename.split('.')[0]
            speaker_name = basedir.split('/')[-3]
            print('%s, %d/%d, speaker %s, sentence %s' % (gender_type, (i + 1), num_sentences, speaker_name, sample_id))

            # output name
            spk = speaker_name[13:]
            mel_filename = spk + '.' + mel_filename

            # rearrange batch data
            x, y = model.parse_batch(batch, use_gpu=False)

            if gender_type == 'degender':
                input_gender_id = tf.random.normal(shape=gender_id.shape, mean=0.5, stddev=0.01)
            else:
                input_gender_id = gender_id

            swap_gender = gender_type == 'invgender'

            # inference
            (mel_padded_out, mel_padded_post_out, speaker_logit_from_mel,
             speaker_logit_from_mel_hidden_text_rate,
             text_hidden, mel_hidden, mel_hidden_text_rate,
             text_logit_from_mel_hidden) = model([mel_padded, text_input_padded,
                                                  mel_lengths, expand_mat_padded, input_gender_id, speaker_id, []],
                                                 input_text, training=False, do_voice_conversion=False,
                                                 use_true_gender_id=use_true_gender_id, swap_gender=swap_gender)

            # # get mel-spec predicted
            post_output1 = mel_padded_post_out.numpy()[0, :, :]
            speaker_id = speaker_id.numpy()  # scalar
            print('elapsed time = %.2f' % (time.time() - tic))
            task = 'TTS' if input_text else 'VC'
            # post_output1 = np.copy(post_output)

            # re-scale and save
            gain = 1.
            mel1 = np.exp(gain * post_output1 * mell_std + mell_mean)
            mellfinal_spk_name = os.path.join(path_save, speaker_name, 'mellinfinal')
            # mellfinal_spk_name = os.path.join(path_save, speaker_name, gender_type, 'mellinfinal')
            os.makedirs(mellfinal_spk_name, exist_ok=True)
            mel_output_path = os.path.join(mellfinal_spk_name, gender_type + '_' + mel_filename)
            # mel_output_path = os.path.join(mellfinal_spk_name, mel_filename)
            data_dict_linear = {'mel': mel1, 'nfft': 2048, 'hoplen': 200, 'winlen': 800,
                                'nmels': 80, 'sr': 16000, 'fmin': 0., 'fmax': int(sr / 2),
                                "time_axis": 1}
            save_var(mel_output_path, data_dict_linear)

            if args.wav:
                # save wav
                data = recover_wav_griffinlim(np.copy(post_output1), n_fft=2048, win_length=800, hop_length=200,
                                              melstd_gain=1.)
                wav_spk_name = os.path.join(path_save, speaker_name, 'audio')
                os.makedirs(wav_spk_name, exist_ok=True)
                wav_output_filename = os.path.join(wav_spk_name,
                                                   gender_type + '_' + mel_filename.replace('mell.npy', 'wav'))
                print('Griffin-Lim audio output: ' + wav_output_filename)
                sndio.write(wav_output_filename, data / np.max(np.abs(data)), rate=sr, format="wav")

            # # compress phone information on recognition encoder
            # compute NORMALIZED compression matrix
            expand_mat = expand_mat_padded.numpy()[0, :, :]
            expand_normalization_vect = np.sum(expand_mat, axis=1, keepdims=True) + 1e-10
            expand_mat = expand_mat / expand_normalization_vect

            # compress text from mel frame rate to text frame rate (deterministic)
            mel_logit_text_rate = expand_mat @ text_logit_from_mel_hidden.numpy()[0, :, :]
            predicted_text = np.argmax(mel_logit_text_rate, axis=1)
            audio_phids = predicted_text
            audio_phids = [id2ph[id] for id in audio_phids]
            # #
            target_text = y[0].numpy()[0, :]
            target_text = [id2ph[int(id)] for id in target_text]
            # #
            # print('Sounds like %s, Decoded text is ' % (id2sp[int(speaker_id)]))
            # #
            # print('predicted')
            # print(audio_phids)
            # print('ground truth')
            # print(target_text)

            err = levenshteinDistance(audio_phids, target_text)
            # print('Levenshtein distance %.2f, number of phones %d' % (err, len(target_text)))

            errs += err
            totalphs += len(target_text)

        print('Mean Levenshtein distance %.2f' % (float(errs) / float(totalphs)))
