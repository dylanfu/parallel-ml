from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import time
from . import model
from . import utils

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.contrib.training.python.training import hparam


def get_args():
    """
    Argument parser.

    Returns:
    Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Training and Eval data local or GCS')
    parser.add_argument(
        '--num-epochs',
        type=float,
        default=5,
        help='number of times to go through the data, default=5')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.001')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    return parser.parse_args()


def train_and_evaluate(params):
    """Helper function: Trains and evaluates model.

    Args:
      params: (dict) Command line parameters passed from task.py
    """
    # Loads data.
    # (train_data, test_data) = \
    #     utils.prepare_data(data_path=params.data_path,
    #                        batch_size=params.batch_size)

    SPLIT_WEIGHTS = (9, 1)
    # splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
    # (train_data, test_data) = tfds.load(name="tf_flowers",
    #                                   split=list(splits),
    #                                   as_supervised=True,
    #                                   data_dir=params.data_path)

    # assert isinstance(train_data, tf.data.Dataset)
    # assert isinstance(test_data, tf.data.Dataset)

    # # Scale values to a range of 0 to 1.
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0

    # Calculating number of images in train and test sets
    # to establish a good step size
    train_steps, test_steps = (
        3670 * weight/10
        for weight in SPLIT_WEIGHTS)

    # num_train, num_test = (
    #     5 * weight/10
    #     for weight in (9,1))
    # train_steps = round(num_train)
    # test_steps = round(num_test)

    # Create TrainSpec.
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: model.input_fn(
            # dataset=train_data,
            batch_size=params.batch_size,
            data_path=params.data_path,
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=train_steps)

    # Create EvalSpec.
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn(
            # dataset=test_data,
            batch_size=params.batch_size,
            data_path=params.data_path,
            mode=tf.estimator.ModeKeys.EVAL),
        # test_data,
        steps=None,
        start_delay_secs=10,
        throttle_secs=10)

    # Define running config.
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
    # Create estimator.
    estimator = model.keras_estimator(
        model_dir=params.job_dir,
        config=run_config,
        learning_rate=params.learning_rate)

    # Start training
    start = time.time()
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    end = time.time()
    print('Time Taken:', end - start)


if __name__ == '__main__':

    args = get_args()
    tf.logging.set_verbosity(args.verbosity)

    params = hparam.HParams(**args.__dict__)
    train_and_evaluate(params)
