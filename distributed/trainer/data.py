import argparse
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
import tensorflow_datasets as tfds


def default_args(argv):
    """Provides default values for Workflow flags."""
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--input_path',
    #     required=True,
    #     help='Input specified as uri to CSV file. Each line of csv file '
    #     'contains colon-separated GCS uri to an image and labels.')
    # parser.add_argument(
    #     '--input_dict',
    #     dest='input_dict',
    #     required=True,
    #     help='Input dictionary. Specified as text file uri. '
    #     'Each line of the file stores one label.')
    parser.add_argument(
        '--output_path',
        required=True,
        help='Output directory to store dataset to.')
    # parser.add_argument(
    #     '--project',
    #     type=str,
    #     help='The cloud project name to be used for running this pipeline')

    # parser.add_argument(
    #     '--job_name',
    #     type=str,
    #     default='flowers-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
    #     help='A unique job identifier.')
    # parser.add_argument(
    #     '--num_workers', default=20, type=int, help='The number of workers.')
    # parser.add_argument('--cloud', default=False, action='store_true')
    # parser.add_argument(
    #     '--runner',
    #     help='See Dataflow runners, may be blocking'
    #     ' or not, on cloud or not, etc.')

    parsed_args, _ = parser.parse_known_args(argv)

    # if parsed_args.cloud:
    #     # Flags which need to be set for cloud runs.
    #     default_values = {
    #         'project':
    #             get_cloud_project(),
    #         'temp_location':
    #             os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
    #         'runner':
    #             'DataflowRunner',
    #         'save_main_session':
    #             True,
    #     }
    # else:
    #     # Flags which need to be set for local runs.
    #     default_values = {
    #         'runner': 'DirectRunner',
    #     }

    # for kk, vv in default_values.iteritems():
    #     if kk not in parsed_args or not vars(parsed_args)[kk]:
    #         vars(parsed_args)[kk] = vv

    return parsed_args


def run(in_args=None):
    SPLIT_WEIGHTS = (9, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
    (raw_train, raw_test) = tfds.load(name="tf_flowers",
                                      split=list(splits),
                                      download=True,
                                      as_supervised=True,
                                      data_dir=in_args.output_path)
    assert isinstance(raw_train, tf.data.Dataset)
    assert isinstance(raw_test, tf.data.Dataset)


def main(argv):
    arg_dict = default_args(argv)
    run(arg_dict)


if __name__ == '__main__':
    main(sys.argv[1:])
