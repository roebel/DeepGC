from hparams import create_hparams, get_root_dir
import os.path
import sys
import random
# set no GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
Generate phone/mel file lists from a subset of the VCTK dataset
Needed to generate lists for inference
"""


def randomfilepairs(file1, file2, file1out, file2out, n_out_line_per_speaker, speaker_list, seed=0):
    with open(file1) as f:
        lines1 = f.readlines()
    with open(file2) as f:
        lines2 = f.readlines()
    random.seed(seed)
    lines1out_list = []
    lines2out_list = []
    for spk in speaker_list:
        spk_lines1subset = [item.strip() for item in lines1 if item.find(spk) > -1]
        indices = random.sample(range(len(spk_lines1subset)), n_out_line_per_speaker)
        count = -1
        for item1, item2 in zip(lines1, lines2):
            if item1.find(spk) == -1:
                continue
            else:
                count += 1
            if count in indices:
                lines1out_list.append(item1.strip())
                lines2out_list.append(item2.strip())
    lines1out = '\n'.join(lines1out_list)
    lines2out = '\n'.join(lines2out_list)
    with open(file1out, 'w') as f:
        f.writelines(lines1out)
    with open(file2out, 'w') as f:
        f.writelines(lines2out)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="generate phone/mel file lists from a subset of the VCTK dataset")
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs (Def: %(default)s)',
                        default=None)
    parser.add_argument('-n', type=int,
                        required=True, help='number of sentences in the output files')
    parser.add_argument('--suffix', type=str,
                        required=True, help='suffix of the output list names')
    parser.add_argument('--dir', type=str, default='list',
                        required=True, help='directory that will contain the output lists')
    parser.add_argument('--set', type=str, default='valid',
                        help="sample from 'train' or 'valid' set (Def: %(default)s")
    parser.add_argument('--db_root_dir', type=str, default=None,
                        help='root dir containing the db_config.py file to be used for training, if not given this will be determined from '
                             'roots.yml located in the same directry as the present script  (Def: read from db_roots.yml)',
                        )
    parser.add_argument("speakers", nargs="+", help="one or more speaker string(s) (like p273 p232 etc)")

    args = parser.parse_args()
    db_root_dir = get_root_dir(config_file=os.path.join(os.path.dirname(__file__), "db_roots.yaml"),
                               dir_tag="root_dir", default_dir=args.db_root_dir)

    hparams = create_hparams(args.hparams, root_dir=db_root_dir)

    print('generate phone/mel file lists for subjective evaluation (subset)')
    # output lists
    os.makedirs(args.dir, exist_ok=True)
    mel_output_list = os.path.join(args.dir, 'mel_' + args.suffix + '.list')
    phone_output_list = os.path.join(args.dir, 'phone_' + args.suffix + '.list')

    # input lists
    root_dir = hparams.root_dir
    if args.set == 'train':
        mel_input_list = os.path.join(root_dir, hparams.mel_training_list_filtered)
        phone_input_list = os.path.join(root_dir, hparams.phone_training_list_filtered)
    elif args.set == 'valid':
        mel_input_list = os.path.join(root_dir, hparams.mel_validation_list)
        phone_input_list = os.path.join(root_dir, hparams.phone_validation_list)
    else:
        sys.exit("unkown set, must be 'train' or 'valid'")

    print('mel input list file: ' + mel_input_list)
    print('mel output list file: ' + mel_output_list)
    print('phone input list file: ' + phone_input_list)
    print('phone output list file: ' + phone_output_list)

    randomfilepairs(mel_input_list, phone_input_list,
                    mel_output_list, phone_output_list, args.n, args.speakers, seed=0)
