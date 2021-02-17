#! /usr/bin/env python

import numpy as np
import librosa
try:
    librosa.feature.melspectrogram(y=np.zeros(2050),norm="slaney")
    librosa_use_norm_slaney = True
except librosa.ParameterError:
    librosa_use_norm_slaney = False

import fileio.iovar as iov
from pysndfile import sndio

log_to_db = 20 * np.log10(np.exp(1))

def calc_melspec(infile, outfile, n_fft, hop_len, win_len, n_mels, fmin, fmax, dtype=np.float32,
                 time_axis=1, log_output=True, log_spec_scale=1, log_spec_offset=0,
                 lin_spec_scale=1, lin_spec_offset=None,  verbose=False, return_results=False):
    """
    Computes mel spectrum from an input audio file and saves it in the output file.
    Uses "librosa" package.

    spectrum scaling will be performed as follows starting with M the linear amplitude of the mel spectrogram
    derived from the signals STFT

    in case of log_output == True

        Mout = log_spec_scale * log(lin_spec_scale * M + lin_spec_offset) + log_spec_offset

    in case of log_output == False

        Mout = lin_spec_scale * M + lin_spec_offset

    Parameters
    ----------
    infile: str
        input file name
    outfile: str
        output file name
    n_fft: int
        fft window size in samples
    hop_len: int
        hop size in samples
    win_len: int
        analysis window size in samples
    n_mels: int
        number of mel bins in the output matrix (n_mels x number of frames)
    fmin : float
        lowest frequency (in Hz)
    fmax : float
        highest frequency (in Hz). If equal to None, use fmax = sample rate / 2.0
    time_axis: int
        the axis for the time dimension. Can be either 0 or 1. Default is 1  so that the resulting melspectrogram willl
        have shape n_mel x n_time.
    log_output: bool
        if set to True the output will be the log amplitude mel spectrum (Def: True)
    lin_spec_scale: Union[float,int]
        scaling factor that will be applied to the linear amplitude mel spectrum. This allows rescaling/normalization of the
        mel spectrum (Def: 1)
    lin_spec_offset: Union[float,int,None]
        offset to be added to the linear amplitude mel spectrum. This allows calculating a
        mel spectrogram of the form log(amp + offset) which avoid exremely small values.
        If set to None the offset will be eps. In case you don't want any offset please explicitly
        set lin_spec_offset to 0. (Def: None)
    log_spec_scale: Union[float,int]
        scaling factor that will be applied to the log amplitude mel spectrum. This allows rescaling/normalization of the
        mel spectrum (Def: 1)
    log_spec_offset: Union[float,int,None]
        offset to be added to the scaled log amplitude mel spectrum. (Def: 0)
    verbose: bool
        display the parameters for the operation to be performed
    Returns
    -------
        nothing, the data is saved in outfile.
    """


    y, sample_rate,_ = sndio.read(infile, dtype=np.dtype(dtype))

    spec = librosa.core.stft(y=y,
                                 n_fft=n_fft,
                                 hop_length=hop_len,
                                 win_length=win_len,
                                 window='hann',
                                 center=True,
                                 pad_mode='reflect')

    spec = librosa.magphase(spec)[0]
    mel_spectrogram = librosa.feature.melspectrogram(S=spec,
                                                     sr=sample_rate,
                                                     n_mels=n_mels,
                                                     power=1.0,  # actually not used given "S=spec"
                                                     fmin=fmin,
                                                     fmax=fmax,
                                                     htk=False,
                                                     norm='slaney' if librosa_use_norm_slaney else 1
                                                     )
    if time_axis == 0:
        mel_spectrogram = mel_spectrogram.T

    if lin_spec_scale != 1:
        mel_spectrogram *= lin_spec_scale
    if lin_spec_offset is not None:
        mel_spectrogram += lin_spec_offset

    data_dict = {'nfft': n_fft, 'hoplen': hop_len, 'winlen': win_len, 'nmels': n_mels,
                 'sr': sample_rate, 'fmin': fmin, 'fmax': fmax,
                 'lin_spec_offset' : lin_spec_offset, 'lin_spec_scale' : lin_spec_scale,
                 'log_spec_offset' : log_spec_offset, 'log_spec_scale' : log_spec_scale,
                 "time_axis": time_axis}

    if log_output:
        if lin_spec_offset is None:
            mel_spectrogram += np.finfo(dtype).eps
        log_mel_spectrogram = log_spec_scale * np.log(mel_spectrogram.astype(dtype=dtype, copy=False)) + log_spec_offset
        # make dict and save data
        data_dict['mell'] =  log_mel_spectrogram
        data_tag = 'mell'
    else:
        data_dict['mel'] =  mel_spectrogram.astype(dtype=dtype, copy=False)
        data_tag = 'mel'

    if return_results:
        return data_dict[data_tag]

    iov.save_var(outfile, data_dict)
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="compute mel spectrogram from raw audio file")

    parser.add_argument("infile", help="input audio file name")
    parser.add_argument("outfile", help="output file name (pickle)")
    parser.add_argument("--fftlen", default=2048, type=int,
                        help="fft window length in samples (Def: %(default)s)")
    parser.add_argument("--wlen", default=800, type=int,
                        help="window length in samples (Def: %(default)s)")
    parser.add_argument("--hlen", default=200, type=int,
                        help="hop length in samples (Def: %(default)s)")
    parser.add_argument("--melchan", default=80, type=int,
                        help="number of mel channels (Def: %(default)s)")
    parser.add_argument("--fmin", default=0., type=float,
                        help="lowest frequency, in Hz, for linear scale to mel scale transform (Def: %(default)s)")
    parser.add_argument("--time_axis", default=1, type=int, choices=[0, 1],
                        help="the axis reflecting time dimension in the output spectrogram. Can be 0 or 1 (Def: %(default)s)")
    parser.add_argument("--fmax", default=None, type=float,
                        help="highest frequency, in Hz, for linear scale to mel scale transform (Def: Nyquist)")
    parser.add_argument("-la", "--lin_amp_output", action="store_true", help="output linear amplitude mel spctrogram (Def: %(default)s)")
    parser.add_argument("--lin_spec_scale", default=1, type=np.float,
                        help="scaling factor to be applied to linear amplitude spectrogram. (Def: %(default)s)")
    parser.add_argument("--log_spec_scale", default=1, type=np.float,
                        help="scaling factor to be applied to log amplitude spectrogram (for normalization). (Def: %(default)s)")
    parser.add_argument("--lin_spec_offset", default=None, type=np.float,
                        help="offset to be applied to linear amplitude spectrogram, can be used to calculate "
                             "log(amp +offset).  If unset this defaults to eps. (Def: %(default)s)")
    parser.add_argument("--log_spec_offset", default=0, type=np.float,
                        help="offset to be applied to log amplitude spectrogram. Can be used to "
                             "center log mel spectum (Def: %(default)s)")
    parser.add_argument("-v", "--verbose",  default=False, action="store_true",
                      help="enable verbose feedback (Def: %(default)s)" )
    parser.add_argument("-p", "--plot", default=False, action="store_true",
                        help="plot results (Def: %(default)s)")

    args = parser.parse_args()
    ret = calc_melspec(infile=args.infile, outfile=args.outfile, n_fft=args.fftlen,
                 hop_len=args.hlen, win_len=args.wlen,
                 n_mels=args.melchan, fmin=args.fmin, fmax=args.fmax,
                 time_axis=args.time_axis,
                 log_output=not args.lin_amp_output,
                 lin_spec_scale=args.lin_spec_scale,
                 lin_spec_offset=args.lin_spec_offset,
                 log_spec_scale=args.log_spec_scale,
                 log_spec_offset=args.log_spec_offset,
                 verbose=args.verbose)

    if args.plot:
        from matplotlib import pypot as plt
        plt.imshow(ret.T)
        plt.title(args.infile)
        plt.colorbar()
        plt.show()
