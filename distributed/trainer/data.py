import argparse
import sys
import tensorflow as tf
import tensorflow_datasets as tfds


def default_args(argv):
    """Provides argument values for Workflow flags."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        required=True,
        help='Output directory to store dataset to.')

    parsed_args, _ = parser.parse_known_args(argv)

    return parsed_args


def run(in_args=None):
    SPLIT_WEIGHTS = (9, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
    (raw_train, raw_test), info = tfds.load(name="tf_flowers",
                                            with_info=True,
                                            split=list(splits),
                                            download=True,
                                            as_supervised=True,
                                            data_dir=in_args.output_path)
    assert isinstance(raw_train, tf.data.Dataset)
    assert isinstance(raw_test, tf.data.Dataset)
    print(info)


def main(argv):
    arg_dict = default_args(argv)
    run(arg_dict)


if __name__ == '__main__':
    main(sys.argv[1:])
