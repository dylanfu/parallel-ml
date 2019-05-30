import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil as ps

#preproccessing

#change the number of hidden layers (1, 4, 8)
hidden = [1, 4, 8]
#change the number of neurons per layer (32, 128, 256)
neuron = [32, 128, 256]
#change the number of epochs (1, 10, 20)
epoch = [1, 10, 20]
#change activation (relu)

#Test accuracy
testAccuracy = []

#Total time
totalTime = []

gogo = ps.cpu_count()
yoyo = ps.cpu_percent(interval=None)
print("###########################")
print(gogo)
print(yoyo)
print("###########################")



fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

models = []

### build the model
for i in range(len(neuron)):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(neuron[i], activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    models.append(model)
###

for i in range(len(models)):

    meep = models[i]

    timeDiff = time.time()

    ###train the model

    gogo = ps.cpu_count()
    yoyo = ps.cpu_percent(interval=0.1)
    print("###########################")
    print(gogo)
    print(yoyo)
    print("###########################")
    meep.fit(train_images, train_labels, epochs=5)
    ###

    timeDiff = time.time() - timeDiff

    ##evaluate the model
    test_loss, test_acc = meep.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    testAccuracy.append(test_acc)

    print('Total Time: ', timeDiff)
    totalTime.append(timeDiff)

print(testAccuracy)
print(totalTime)


