import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 5
IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 32

"""
Format given dataset
"""
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

"""
Augments given dataset
"""
def augment_data(image, label):
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
  # Maybe add more augmentation
  return image, label

"""
Creating a simple CNN model in keras using functional API
"""
def create_model():
    img_inputs = keras.Input(shape=IMG_SHAPE)
    conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(img_inputs)
    maxpool_1 = keras.layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1)
    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)
    flatten = keras.layers.Flatten()(conv_3)
    dense_1 = keras.layers.Dense(64, activation='relu')(flatten)
    output = keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')(dense_1)

    model = keras.Model(inputs=img_inputs, outputs=output)
    
    return model

"""
Train given model using dataset
"""
def train_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train.repeat(),
              epochs=EPOCHS, 
              steps_per_epoch=steps_per_epoch,
              validation_data=validation.repeat(),
              validation_steps=validation_steps)
    
    return history

"""
Evaluates given model
"""
def test_model(model):
    test_loss, test_acc = model.evaluate(test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    predictions = model.predict(test)

    return predictions

# Load dataset from tensorflow dataset API
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="tf_flowers", 
                                                            with_info=True,
                                                            split=list(splits),                                                            
                                                            as_supervised=True)

# Format datasets
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Pre-process dataset
train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation = validation.batch(BATCH_SIZE)
test = test.batch(BATCH_SIZE)
# (Optional) prefetch will enable the input pipeline to asynchronously fetch batches while
# your model is training.
train = train.prefetch(tf.data.experimental.AUTOTUNE)

# Augment training dataset
train = train.map(augment_data)

# Calculating number of images in train, val and test sets
# to establish a good batch size
num_train, num_val, num_test = (
metadata.splits['train'].num_examples * weight/10 
for weight in SPLIT_WEIGHTS
)
steps_per_epoch = round(num_train) #BATCH_SIZE
validation_steps = round(num_val) #BATCH_SIZE

# Generate model
model = create_model()

# Train model and get training metrics
history = train_model(model)

# Evaluate model and get prediction metrics
prediction = test_model(model)