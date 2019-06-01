# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import subprocess
import tensorflow as tf
import tensorflow_datasets as tfds


WORKING_DIR = os.getcwd()
LOCAL_PATH = 'flower-data'
IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SHUFFLE_BUFFER_SIZE = 1024


def download_files_from_gcs(source, destination):
    """Download files from GCS to a WORKING_DIR/.

    Args:
      source: GCS path to the training data
      destination: GCS path to the validation data.

    Returns:
      A list to the local data paths where the data is downloaded.
    """
    local_file_names = [destination]
    gcs_input_paths = [source]

    # Copy raw files from GCS into local path.
    raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
                                  for local_file_name in local_file_names
                                  ]
    for i, gcs_input_path in enumerate(gcs_input_paths):
        if gcs_input_path:
            subprocess.check_call(
                ['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

    return raw_local_files_data_paths


def _load_data(path, destination):
    """Verifies if file is in Google Cloud.

    Args:
      path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
      destination: (str) The filename to save as on local disk.

    Returns:
      A filename
    """
    if path.startswith('gs://'):
        download_files_from_gcs(path, destination=destination)
        return destination
    return path


def format_example(image, label):
    """
    Format given dataset
    """
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def augment_data(image, label):
    """
    Augments given dataset
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    # Maybe add more augmentation
    return image, label


def prepare_data(data_path, batch_size):
    """Loads tf flowers files.

    Args:
      data_path: (str) Location where data files are located.

    Returns:
      A tuple of training and test data.
    """
#   (train_data, test_data)= _load_data(data_path, LOCAL_PATH)
    SPLIT_WEIGHTS = (9, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
    (raw_train, raw_test) = tfds.load(name="tf_flowers",
                                      split=list(splits),
                                      as_supervised=True,
                                      data_dir=data_path)
    
    # # Format datasets
    # train = raw_train.map(format_example)
    # test = raw_test.map(format_example)

    # # Pre-process dataset
    # train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    # test = test.batch(batch_size)
    # # (Optional) prefetch will enable the input pipeline to asynchronously fetch
    # # batches while your model is training.
    # # train = train.prefetch(tf.data.experimental.AUTOTUNE)

    # # Augment training dataset
    # train = train.map(augment_data)

    return (raw_train, raw_test)
