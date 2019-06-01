from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.python.keras import models
from tensorflow import keras

tf.logging.set_verbosity(tf.logging.DEBUG)


def keras_estimator(model_dir, config, learning_rate):
    """Creates a Keras model with layers.

    Args:
      model_dir: (str) file path where training files will be written.
      config: (tf.estimator.RunConfig) Configuration options to save model.
      learning_rate: (int) Learning rate.

    Returns:
      A keras.Model
    """
#   model = models.Sequential()

#   model.add(Flatten(input_shape=(28, 28)))
#   model.add(Dense(128, activation=tf.nn.relu))
#   model.add(Dense(10, activation=tf.nn.softmax))

    img_inputs = keras.Input(shape=(64, 64, 3))
    conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(img_inputs)
    maxpool_1 = keras.layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1)
    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)
    flatten = keras.layers.Flatten()(conv_3)
    dense_1 = keras.layers.Dense(64, activation='relu')(flatten)
    output = keras.layers.Dense(5, activation='softmax')(dense_1)

    model = keras.Model(inputs=img_inputs, outputs=output)

    # Compile model with learning parameters.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator


def input_fn(batch_size, data_path, mode):
    """Input function.

    Args:
      features: (numpy.array) Training or eval data.
      labels: (numpy.array) Labels for training or eval data.
      batch_size: (int)
      mode: tf.estimator.ModeKeys mode

    Returns:
      A tf.estimator.
    """
    # Default settings for training.
    # if labels is None:
    #     inputs = features
    # else:
    #     # Change numpy array shape.
    #     inputs = (features, labels)
    # Convert the inputs to a Dataset.
    # dataset = tf.data.Dataset.from_tensor_slices(inputs)

    SPLIT_WEIGHTS = (9, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
    (train_data, test_data), info = tfds.load(name="tf_flowers",
                                              with_info=True,
                                              download=True,
                                              split=list(splits),
                                              as_supervised=True,
                                              data_dir=data_path)

    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(test_data, tf.data.Dataset)
    # assert isinstance(dataset, tf.data.Dataset)

    print(info)

    # Format datasets
    # dataset = dataset.map(format_example)
    # test = raw_test.map(format_example)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = train_data.map(format_example)
        dataset = dataset.shuffle(1024).batch(batch_size)
        dataset = dataset.map(augment_data)
        dataset = dataset.repeat()
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        dataset = test_data.map(format_example)
        dataset = dataset.batch(batch_size)

    # return dataset.make_one_shot_iterator().get_next()
    return dataset


def format_example(image, label):
    """
    Format given dataset
    """
    # image = tf.squeeze(image, squeeze_dims=[5])
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (64, 64))
    return image, label[np.newaxis]


def augment_data(image, label):
    """
    Augments given dataset
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    # Maybe add more augmentation
    return image, label[:, tf.newaxis]


def serving_input_fn():
    """Defines the features to be passed to the model during inference.

    Expects already tokenized and padded representation of sentences

    Returns:
      A tf.estimator.export.ServingInputReceiver
    """
    feature_placeholder = tf.placeholder(tf.float32, shape=(64, 64))
    features = feature_placeholder
    return tf.estimator.export.TensorServingInputReceiver(features,
                                                          feature_placeholder)
