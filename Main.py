# Needs the directory of data (containing .matrix , .bed files for interactions and ranges) as the 1st argument
# the save directory as the 2nd argument
import os
from sys import argv
from sys import path as sys_path
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
from os import listdir, path
from sys import stderr, exit, stdout
from tensorflow.python.client import device_lib


def clip_slash_from_end(x):
    if x.endswith('/'):
        return x[:-1]
    elif x.endswith('\\'):
        return x[:-1]
    else:
        return x


def define_arg_parser():
    parser = argparse.ArgumentParser(description='Calculating significance of Hi-C interactions.')

    parser.add_argument('base_dir',
                        type=clip_slash_from_end,
                        help='A directory containing exatly one .matrix file for interactions and one .bed file for bins.',
                        metavar='base_directory')

    parser.add_argument('save_dir',
                        type=clip_slash_from_end,
                        help='A directory for saving the results in it.',
                        metavar='save_directory')

    # Tool-related arguments

    parser.add_argument('-d', '--device', dest='device',
                        default='CPU:0', type=str,
                        help='The device that will be used for training the model.',
                        metavar='device')

    parser.add_argument('-t', '--threads', dest='threads_num',
                        default=24, type=int,
                        help='The number of CPU threads to use for training the model, in the case of using CPU.',
                        metavar='Threads_number')

    parser.add_argument('-s', '--silent', dest='silent_mode',
                        default=True, type=(lambda x: not x.lower().startswith('f')),
                        help='Only very necessary information would be printed.',
                        metavar='silent_model')

    parser.add_argument('-do', '--detailed_output', dest='detailed_output',
                        default=False, type=(lambda x: not x.lower().startswith('f')),
                        help='If True, a more detailed output will be produced.',
                        metavar='detailed_output')

    # Training related arguments

    parser.add_argument('-pvl', '--pval_limit', dest='p_val_limit',
                        default=0.001, type=float,
                        help='The p-value level that the interactions satisfying it would be considered significant and would not be included in training the model.',
                        metavar='significance_limit')

    parser.add_argument('-r', '--rounds', dest='rounds_num',
                        default=4, type=int,
                        help='The number of rounds for training the model without the interactions identified as significant.',
                        metavar='Training_rounds')

    parser.add_argument('-rs', '--replace_significants', dest='replace_significants_in_bias_calculation',
                        default=True, type=(lambda x: not x.lower().startswith('f')),
                        help='If true, the expected read count of the significant interactions would be considered in calculating bias factor of bins instead of the observed read count.',
                        metavar='Replacing_significants')

    parser.add_argument('-mind', '--min_distance', dest='min_distance',
                        default=0, type=int,
                        help='Interactions with less distance would be excluded from training the model.',
                        metavar='Minimum_distance')

    parser.add_argument('-maxd', '--max_distance', dest='max_distance',
                        default=-1, type=int,
                        help='Interactions with more distance would be excluded from training the model, -1 means there is no maximum limit.',
                        metavar='Maximum_distance')

    parser.add_argument('-minr', '--min_read', dest='min_read',
                        default=1, type=int,
                        help='Interactions with fewer read count would be excluded from training the model.',
                        metavar='Minimum_read_count')

    # capture related arguments

    parser.add_argument('-c', '--capture', dest='run_capture',
                        default=False, type=(lambda x: not x.lower().startswith('f')),
                        help='Whether the capture version must be run (vs general).',
                        metavar='run_capture')

    parser.add_argument('-brl', '--bait_ratio_lim', dest='bait_ratio_lim',
                        default=0.0, type=float,
                        help='The minimum required overlap a bin must have with target regions with respect to its length to be considered as bait.',
                        metavar='bait_ratio_limit')

    parser.add_argument('-bll', '--bait_len_lim', dest='bait_len_lim',
                        default=1, type=int,
                        help='The minimum required overlap a bin must have with target regions to be considered as bait.',
                        metavar='bait_length_limit')

    parser.add_argument('-bo', '--bait_overhangs', dest='bait_overhangs',
                        default=0, type=int,
                        help='The extra number of base-pairs from each side of a target region that will also be considered as target region.',
                        metavar='bait_overhangs')

    parser.add_argument('-bd', '--baits_dir', dest='baits_dir',
                        default='', type=str,
                        help='The directory of the file containing information about target regions.',
                        metavar='baits_directory')

    parser.add_argument("-v", "--version", action="version", version="MAXHiC 1.0"
                        , help="Prints version and exit")

    return parser


def check_validity_of_base_dir(parser):

    base_dir = parser.base_dir

    matrix_dir = None
    bed_dir = None

    for f in listdir(base_dir):

        if f.endswith('.matrix'):
            if matrix_dir is None:
                matrix_dir = 0
            else:
                print('Error: Multiple .matrix files in the given directory', file=stderr)
                return -1
        elif f.endswith('.bed'):
            if bed_dir is None:
                bed_dir = 0
            else:
                print('Error: Multiple .bed files in the given directory', file=stderr)
                return -1

    if matrix_dir is None:
        print('Error: No .matrix file in the given directory', file=stderr)
        return -1

    if bed_dir is None:
        print('Error: No .bed file in the given directory', file=stderr)
        return -1

    return 0


def check_baits_if_capture(parser):

    if parser.run_capture:
        if parser.baits_dir == '':
            print('Error: --baits_dir argument is required for the capture version.', file=stderr)
            return -1
        elif not path.exists(parser.baits_dir):
            print('The given file for targets regions does not exist:\n' + parser.baits_dir, file=stderr)
            return -1

    return 0


def check_device(parser):

    local_device_protos = device_lib.list_local_devices()
    local_devices_names = [x.name for x in local_device_protos]

    local_devices_names = [local_devices_names[i] for i in (np.argsort([len(x) for x in local_devices_names])).tolist()]

    found_device = None
    for ldn in local_devices_names:
        if parser.device.lower() in ldn.lower():
            found_device = ldn
            break

    if found_device is None:
        print('%s not available as a device. The list of available devices:' % parser.device)
        print(', '.join(local_devices_names))
        return -1

    return found_device


def run_suitable_script(parser):

    path_from_cwd_to_main = os.path.abspath(argv[0])
    if not parser.run_capture:
        sys_path.insert(0, os.path.dirname(path_from_cwd_to_main) + '/General')
        from General.Main import general_main
        general_main(parser.base_dir, parser.save_dir,
                     parser.p_val_limit, parser.threads_num, parser.replace_significants_in_bias_calculation, parser.rounds_num,
                     parser.min_distance, parser.max_distance, parser.min_read, parser.device, parser.silent_mode, parser.detailed_output)
    else:
        sys_path.insert(0, os.path.dirname(path_from_cwd_to_main) + '/Capture')
        from Capture.Main import capture_main
        capture_main(parser.base_dir, parser.save_dir,
                     parser.p_val_limit, parser.threads_num, parser.replace_significants_in_bias_calculation, parser.rounds_num,
                     parser.min_distance, parser.max_distance, parser.min_read, parser.device, parser.silent_mode, parser.detailed_output,
                     parser.baits_dir, parser.bait_ratio_lim, parser.bait_len_lim, parser.bait_overhangs)


if __name__ == '__main__':

    the_parser = define_arg_parser()
    args = the_parser.parse_args()

    if check_validity_of_base_dir(args) != 0:
        exit(-1)

    if check_baits_if_capture(args) != 0:
        exit(-1)

    full_device_name = check_device(args)
    if full_device_name == -1:
        exit(-1)
    args.device = full_device_name

    run_suitable_script(args)
