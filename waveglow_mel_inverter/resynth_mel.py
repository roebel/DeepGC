#!/usr/bin/env python3
# coding: utf-8

import os, sys

# support relative imports up to the top level directory
path_components = [ pp for pp in os.path.dirname(os.path.abspath(__file__)).split('/') if pp != "."]
__package__ = ".".join(path_components[-2:])
print(__package__)

# silence verbose TF feedback
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


import numpy as np
import tensorflow as tf
import time
from pysndfile import sndio
from waveglow_model.waveglow_multi_res import WaveGlow_MR
from waveglow_model import training_utils as utils
from ..gender_converter.fileio import iovar as iov
from copy import deepcopy

log_to_db = 20 * np.log10(np.exp(1))

def pad_to(insound, segment_length, offset=0):
    insize = insound.size + offset
    if segment_length * (insize // segment_length) < insound.size :
        insound = np.pad(insound, pad_width=((offset, segment_length *(1+ (insize // segment_length)) - insize),),
                                      mode="constant")
    elif offset:
        insound = np.pad(insound, pad_width=((offset, offset),),
                                      mode="constant")

    return insound
    
def main(model_dir, input_mell_files, output_dir,
         single_seg_synth=True, use_gpu=False, sigma=1.,
         format="flac", verbose=False, seed=42):

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "model_dir")

    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_file) :
        raise FileNotFoundError(f"error::loading config file from {config_file}")

    if verbose:
        print(f"read config  from {config_file}", file=sys.stderr)
    hparams = utils.read_config(config_file=config_file)
    training_config = hparams['training_config']
    preprocess_config = hparams['preprocess_config']

    # ## Instantiate model and optimizer
    myWaveGlow = WaveGlow_MR(waveglow_config=hparams['waveglow_mr_config'],
                                 training_config=training_config,
                                 preprocess_config=preprocess_config,
                                 name='myWaveGlow')

    # we need to run the model at least once si that all componnts are built ogherwise the
    # state that is loaded from the checkpoint will disappear once the model is run
    # the first time when all layers are built.
    segment_length = preprocess_config['segment_length']
    if segment_length != myWaveGlow.segment_length:
        raise RuntimeError(f"segment length of config file {segment_length} and waveglow model {myWaveGlow.segment_length} does not match")
    myWaveGlow.build_model()

    model_weights_path = os.path.join(model_dir, "weights.h5")
    if verbose:
        print(f"restore from {model_weights_path}", file=sys.stderr)

    myWaveGlow.load_weights(model_weights_path)
    # seed random number generators
    if seed >= 0:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hop_size = preprocess_config["hop_size"]
    win_len = preprocess_config["fft_size"]
    fmin = preprocess_config["fmin"]
    fmax = preprocess_config["fmax"]
    srate = preprocess_config['sample_rate']
    if 'win_size' in preprocess_config:
        win_len = preprocess_config['win_size']

    preprocess_config_eval = deepcopy(preprocess_config)
    preprocess_config_eval["lin_amp_scale"] = 1
    preprocess_config_eval["lin_amp_off"] = 1e-5
    preprocess_config_eval["mel_amp_scale"] = 1

    lin_amp_scale = 1
    if ("lin_amp_scale" in preprocess_config) and (preprocess_config["lin_amp_scale"] != 1):
         lin_amp_scale = preprocess_config["lin_amp_scale"]

    lin_amp_off = 1.e-5
    if "lin_amp_off" in preprocess_config and (preprocess_config["lin_amp_off"] is not None):
        lin_amp_off = preprocess_config["lin_amp_off"]

    mel_amp_scale = 1
    if ("mel_amp_scale" in preprocess_config) and (preprocess_config["mel_amp_scale"] != 1):
        mel_amp_scale = preprocess_config["mel_amp_scale"]

    use_max_limit = False
    if "use_max_limit" in preprocess_config and preprocess_config["use_max_limit"]:
        use_max_limit = preprocess_config["use_max_limit"]

    for mell_file in input_mell_files:
        if verbose:
            print(f"load mell  from {mell_file}", file=sys.stderr)
        dd = iov.load_var(mell_file)
        if dd['winlen'] != win_len:
            raise RuntimeError(f"mell window size {dd['winlen']} does not match model window size {win_len}")
        if dd['hoplen'] != hop_size:
            raise RuntimeError(f"mell hop_size {dd['hoplen']} does not match model hop size {hop_size}")
        if dd['fmin'] != fmin:
            raise RuntimeError(f"mell fmin {dd['fmin']} does not match model fmin {fmin}")
        if ((dd['fmax'] is None) and fmax !=  dd['sr']/2) or ((dd['fmax'] is not None) and dd['fmax'] != fmax):
            raise RuntimeError(f"mell fmax {dd['fmax']} does not match model fmax {fmax}")
        if dd['sr'] != srate:
            raise RuntimeError(f"mell srate {dd['sr']} does not fit model sr {srate}")
        
        if "mell" in dd:
            log_mel_spectrogram = dd['mell'].T[np.newaxis]
            if "log_spec_offset" in dd and dd["log_spec_offset"] != 0:
                log_mel_spectrogram -= dd["log_spec_offset"]
            if "log_spec_scale" in dd and dd["log_spec_scale"] != 1:
                log_mel_spectrogram /= dd["log_spec_scale"]
            mel_spectrogram = np.exp(log_mel_spectrogram)
        elif "mel" in dd:
            mel_spectrogram = np.array(dd['mel'].T[np.newaxis])
        else:
            raise RuntimeError(f"error::no supported mel spectrum (keys:mell or mell) in {mell_file}")

        if ("lin_spec_offset" in dd) and (dd["lin_spec_offset"] is not None) and (dd["lin_spec_offset"] != 0):
            mel_spectrogram -= dd["lin_spec_offset"]
        if "lin_spec_scale" in dd and dd["lin_spec_scale"] != 1:
            mel_spectrogram /= dd["lin_spec_scale"]       

        if lin_amp_scale != 1:
            mel_spectrogram *= lin_amp_scale

        if use_max_limit:
            mell = np.log(np.fmax(mel_spectrogram, lin_amp_off)).astype(np.float32)
        else:
            mell = np.log(mel_spectrogram + lin_amp_off).astype(np.float32)

        if verbose :
            print(f"stats conditioning mell:: mean: {log_to_db * np.mean(mell):.3f}dB, median: {log_to_db * np.median(mell):.3f}dB, max: {log_to_db * np.max(mell):.3f}dB, min: {log_to_db * np.min(mell):.3f}dB mell.shape {mell.shape}")
            
        ori_size = mell.shape[1] * hop_size
        synth_len = ori_size
        start_time = time.time()
        syn_audio = myWaveGlow.infer(mel_amp_scale * mell, sigma=sigma, synth_length=synth_len).numpy()
        syn_audio = syn_audio.ravel()
        end_time = time.time()

        if verbose:
            print(f"synthesized audio with {syn_audio.size} samples "
                  f"in {end_time-start_time:.3f}s ({syn_audio.size/(end_time-start_time):.2f}Hz)", file=sys.stderr)

        outfile = os.path.join(output_dir, "syn_" + os.path.splitext(os.path.basename(mell_file))[0]+"."+format)
        if np.max(np.abs(syn_audio[:ori_size])) > 1:
            norm = 0.99/np.max(np.abs(syn_audio))
            print(f'to prevent clipping you would need to normalize {outfile} by {norm:.3f}')

        if verbose:
            print(f"save audio under {outfile}", file=sys.stderr)
        sndio.write(outfile, data=syn_audio[:ori_size], rate=preprocess_config['sample_rate'], format=format)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description="pass a given audio file through and analysis/resynthesis cycle using a waveglow model")
    parser.add_argument("--model_dir", default=None, help=("directory containg weights.h5 and config.yam of the model to be used. "
                                                           f"(Def: {os.path.join(os.path.dirname(__file__), 'model_dir')}"))
    parser.add_argument("-i", "--input_mell_files", nargs="+", required=True, help="list of mell spectra stored in pickle files")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory where synthetic sounds will be stored")
    parser.add_argument("--sigma", default=1, type=np.float32, help="sigma to be used for generating the internal state of the generator (Def: %(default)s)")
    parser.add_argument("--format", default="flac", help="file format for generated audio files (Def: %(default)s)")
    parser.add_argument("--seed", default=42, type=np.int32, help="seed value for random generator (Def: %(default)s)")

    parser.add_argument("-g", "--use_gpu", action="store_true", help="run on gpu")
    parser.add_argument("-v", "--verbose", action="store_true", help="display verbose progress info")

    args= parser.parse_args()
    main(**vars(args))
