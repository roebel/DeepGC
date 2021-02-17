"""
The correspondence between the lab files and the mel spectrogram files is implicit.
The lab file list and the mel-spec file list are provided and the entries (lines) correspond
because they are sorted.
"""
import random
import numpy as np
import tensorflow as tf
from fileio.iovar import load_var
import os.path
from .symbols import get_experiment_speaker_info, get_experiment_phn_info
# seen_speakers is a global variable
from debugprint import print_debug
from tensorflow.keras.utils import Sequence


def read_text(fn):
    '''
    read phone alignments from file of the format:
    start end phone
    '''
    text = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        start, end, phone = line.strip().split()
        text.append([start, end, phone])
    return text


def read_text_for_expandmat(fn):
    '''
    read phone alignments from file of the format:
    start end phone
    input:
        fn: input file
    output:
        timestart: list of start timestamps
        timeend: list of end timestamps
        text: list of phonemes
    '''
    text = []
    timestart = []
    timeend = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        start, end, phone = line.strip().split(' ')
        text.append(phone)
        timestart.append(float(start))
        timeend.append(float(end))
    return timestart, timeend, text


class TextMelIDLoader2():

    def __init__(self, root, mel_list_file, mel_mean_std_file, lab_list_file, shuffle=False, verbose=False,
                 mel_phone_lists_relative_to_root=True):
        '''
        ### NO: list_file: 3-column: (path, n_frames, n_phones)

        root is the root dir for the list files and mean/std file

        list_file: 1-column: path
        mean_std_file: tensor loadable into numpy, of shape (2, feat_dims), i.e. [mean_row, std_row]
        '''

        self.seen_speakers, self.sp2id, _ = get_experiment_speaker_info(root)
        self.phone_list, self.ph2id, _, = get_experiment_phn_info()

        self.verbose = verbose
        if mel_phone_lists_relative_to_root:
            full_mel_list_file = os.path.join(root, mel_list_file)
        else:
            # in inference use local list files
            full_mel_list_file = mel_list_file
        with open(full_mel_list_file) as f:
            lines = f.readlines()
        mel_file_list = [line.strip() for line in lines]

        if mel_phone_lists_relative_to_root:
            full_lab_list_file = os.path.join(root, lab_list_file)
        else:
            # in inference use local list files
            full_lab_list_file = lab_list_file
        with open(full_lab_list_file) as f:
            lines = f.readlines()
        lab_file_list = [line.strip() for line in lines]

        speaker_id = [item for line in lab_file_list for item in line.split('/')
                      if item in self.seen_speakers]

        self.speaker_id = speaker_id
        self.mel_file_list = mel_file_list
        self.lab_file_list = lab_file_list
        data_dict = load_var(os.path.join(root, mel_mean_std_file))
        self.mel_mean = data_dict['mean_mell']
        self.mel_std = data_dict['std_mell']

        # self.onehot = np.eye(len(self.phone_list))

    def get_text_mel_id_tuple(self, idx):
        '''
        read our own data.

        Returns:

        object: dimensionality
        -----------------------
        text_input: [len_text]
        mel: [mel_bin, len_mel]
        spc: [spc_bin, len_spc] => REMOVED
        speaker_id: [1]
        '''

        # read phone list of a sentence
        lab_filename = self.lab_file_list[idx]
        text_input_phonelevel = self.get_text(lab_filename)

        # load mel spectrogram
        mel_filename = self.mel_file_list[idx]
        if self.verbose:
            print(mel_filename)
        mel_input = load_var(mel_filename)['mell']
        # Normalize audio
        mel_input = (mel_input - self.mel_mean) / self.mel_std
        
        # expand_mat = load_var(mat_filename)['expandmat']  # N phone frames x N mel frames

        # compute "expand_mat" ON THE FLY TO AVOID COMPUTING expand_mat in the database
        timestart, timeend, text = read_text_for_expandmat(lab_filename)
        # end time in input lab units
        Tphone = timeend[-1]
        # read mell data only to get the number of frames
        Tmel = mel_input.shape[1]

        # convert timestamps from lab units to frame units (np int16 arrays)
        timestart_mel = (np.asarray(timestart)/Tphone*Tmel).astype(np.int16)
        timeend_mel = (np.asarray(timeend)/Tphone*Tmel).astype(np.int16)

        expand_mat = np.zeros((len(text), Tmel))
        # loop on phones
        tbeg = 0
        for counter0, ph in enumerate(text):
            # duplicate the phone ph
            # tbeg = timestart_mel[counter0]
            tend = timeend_mel[counter0]
            expand_mat[counter0, tbeg:tend] = 1.
            tbeg = tend
        if tend < Tmel:
            expand_mat[counter0, tend:Tmel] = 1.

        # compute input text at mel-spec level
        text_input_mellevel = []
        for ind_t in range(mel_input.shape[1]):
            # get no of phone in the sentence for a given frame
            ind_ph_t = np.argmax(expand_mat[:, ind_t])
            phid_t = text_input_phonelevel[ind_ph_t]
            text_input_mellevel.append(phid_t)  # dilated text

        # get speaker id
        speaker_id = self.speaker_id[idx]
        if speaker_id.find('female') > -1:
            gender_id = np.array([0], dtype=np.int16)
        else:
            gender_id = np.array([1], dtype=np.int16)
        # do not format numpy data

        # # Format for pytorch
        text_input_phonelevel = np.array(text_input_phonelevel, dtype=np.int16)
        text_input_mellevel = np.array(text_input_mellevel, dtype=np.int16)
        speaker_id = np.array([self.sp2id[speaker_id]], dtype=np.int16)

        # dirty hack
        mat_onehot_mellevel = gender_id

        return (text_input_phonelevel, text_input_mellevel, mel_input, mat_onehot_mellevel,
                expand_mat, speaker_id)

    def get_text(self, text_path):
        '''
        Returns:

        text_input: a list of phoneme IDs corresponding
        to the transcript of one utterance
        '''
        text = read_text(text_path)
        text_input = []

        for start, end, ph in text:
            text_input.append(self.ph2id[ph])

        return text_input

    def __getitem__(self, index):
        return self.get_text_mel_id_tuple(index)

    def __len__(self):
        return len(self.mel_file_list)


class myDataLoader2(Sequence):

    def __init__(self, data_set, batch_size):
        self.data_set = data_set
        self.batch_size = batch_size
        self.set_len = len(data_set)
        self.indices = range(self.set_len)

    def randomize(self):
        # permute data item indices
        # do this at the BEGINING OF EACH EPOCH
        self.indices = random.sample(range(self.set_len), k=self.set_len)

    def __len__(self):
        return len(self.data_set) // self.batch_size

    def __getitem__(self, idx):
        # get (permuted) batch

        # create batch (append data)
        batch_x = []
        batch_size = self.batch_size
        for i in range(batch_size):
            batch_x.append(self.data_set[self.indices[idx*batch_size+i]])

        # get text/mel lengths
        text_lengths = [x[4].shape[0] for x in batch_x]
        mel_lengths = [x[2].shape[1] for x in batch_x]

        # TextMelIDLoader output shoudl be :
        # (text_input_phonelevel, text_input_mellevel, mel_input, mat_onehot_mellevel, expand_mat, speaker_id)
        # get max length
        max_text_len = np.max(text_lengths)
        max_mel_len = np.max(mel_lengths)

        mel_bin = batch_x[0][2].shape[0]

        text_input_phonelevel_padded = np.zeros((batch_size, max_text_len), dtype=np.int16)
        text_input_padded = np.zeros((batch_size, max_mel_len), dtype=np.int16)
        mel_padded = np.zeros((batch_size, mel_bin, max_mel_len), dtype=np.float32)
        expand_mat_padded = np.zeros((batch_size, max_text_len, max_mel_len), dtype=np.float32)
        speaker_id = np.zeros((batch_size,), dtype=np.int16)
        gender_id = np.zeros((batch_size,), dtype=np.int16)

        for i in range(batch_size):
            text_ph = batch_x[i][0]
            text_mel = batch_x[i][1]
            mel = batch_x[i][2]

            expandmat = batch_x[i][4]

            text_input_phonelevel_padded[i, :text_ph.shape[0]] = text_ph
            text_input_padded[i, :text_mel.shape[0]] = text_mel
            mel_padded[i, :, :mel.shape[1]] = mel
            expand_mat_padded[i, :text_ph.shape[0], :mel.shape[1]] = expandmat
            speaker_id[i] = batch_x[i][5][0]
            gender_id[i] = batch_x[i][3][0]

        # convert numpy data to tf tensor
        tf_text_input_phonelevel_padded = tf.convert_to_tensor(text_input_phonelevel_padded,
                                                               dtype=tf.dtypes.int16)
        tf_text_input_padded = tf.convert_to_tensor(text_input_padded, dtype=tf.dtypes.int16)
        tf_mel_padded = tf.convert_to_tensor(mel_padded, dtype=tf.dtypes.float32)
        tf_expand_mat_padded = tf.convert_to_tensor(expand_mat_padded, dtype=tf.dtypes.float32)
        tf_speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.dtypes.int16)
        tf_gender_id = tf.convert_to_tensor(gender_id, dtype=tf.dtypes.int16)

        tf_mel_lengths = tf.expand_dims(tf.convert_to_tensor(mel_lengths), axis=1)
        tf_text_lengths = tf.expand_dims(tf.convert_to_tensor(text_lengths), axis=1)

        # UNUSED
        tf_stop_token_padded = []
        tf_mat_onehot_padded = tf_gender_id  # dirty hack!

        return (tf_text_input_phonelevel_padded, tf_text_input_padded, tf_mel_padded, tf_mat_onehot_padded,
                tf_expand_mat_padded, tf_speaker_id, tf_text_lengths, tf_mel_lengths,
                tf_stop_token_padded)
