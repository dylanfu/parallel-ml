import tensorflow as tf
from tensorflow import keras
import statistics as st
import time
import psutil
from multiprocessing import Process, Queue, Value, Lock



### Helper methods

def switch(num):
    if num.value == 0:
        num.value = 1
    else:
        num.value = 0

##getting core information
def f(q, num, lock):
    local_cores = psutil.cpu_count()
    local_int = num.value
    core_usage = []

    for i in range(local_cores):
        core_usage.append([])

    while(True):
        if num.value == local_int:
            hold_cpu_values = psutil.cpu_percent(None, True)

            for x in range(local_cores):
                core_usage[x].append(hold_cpu_values[x])


        else:
            lock.acquire()
            local_int = num.value
            try:
                q.put(getAvg(core_usage))
            except:
                q.put("blank message to be discarded")
            core_usage.clear()

            for i in range(local_cores):
                core_usage.append([])

            lock.release()

        if(num.value != 3):
            time.sleep(1)
        else:
            return

##averaging the core information
def getAvg(array):
    hold_averages = []
    for y in range(len(array)):
        hold_averages.append(st.mean(array[y]))

    return hold_averages


#creating models that number of cores only
def test1(num, q):
    #set variables
    hidden_layers = 1
    neuron_per_layer = 128
    num_of_epochs = 5
    cores_to_test = 3

    models.clear()

    ### build the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ###
    switch(num)
    q.get()
    hold_core = []
    for x in range(cores_to_test +1):
        hold_core.append(x)

    ps.cpu_affinity(hold_core)

    time_diff = time.time()

    ###train the model
    model.fit(train_images, train_labels, epochs=num_of_epochs)
    ###

    time_diff = time.time() - time_diff
    switch(num)
    core_usage = q.get()

    ##get all information needed from model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    info = []
    info.append(test_loss)
    info.append(test_acc)
    info.append(len(hold_core))
    info.append(time_diff)
    info.append(num_of_epochs)
    info.append(hidden_layers)
    info.append(neuron_per_layer)

    title = "Test1"

    outputAppend(info, core_usage, title)


# creating models with different neurons only
def test2(num, q):
    models.clear()

    # set variables
    hidden_layers = 1

    neuron = [32, 128, 256, 512]

    ### build the models
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

    hold_core = []
    for x in range(cores):
        hold_core.append(x)

    ps.cpu_affinity(hold_core)

    for i in range(len(models)):
        switch(num)
        q.get()


        time_diff = time.time()

        ###train the model
        models[i].fit(train_images, train_labels, epochs=num_of_epochs)
        ###

        time_diff = time.time() - time_diff
        switch(num)
        core_usage = q.get()

        ##get all information needed from model
        test_loss, test_acc = models[i].evaluate(test_images, test_labels)

        info = []
        info.append(test_loss)
        info.append(test_acc)
        info.append(len(hold_core))
        info.append(time_diff)
        info.append(num_of_epochs)
        info.append(hidden_layers)
        info.append(neuron[i])

        if i == 0:
            title = "Test2"
        else:
            title = ""

        outputAppend(info, core_usage, title)


# creating models that change the number of hidden layers only
def test3(num, q):
    # set variables
    neuron_per_layer = 128

    models.clear()

    ### build the models
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    models.append(model)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    models.append(model)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    models.append(model)

    hold_core = []
    for x in range(cores):
        hold_core.append(x)

    ps.cpu_affinity(hold_core)

    ###
    for i in range(len(models)):
        switch(num)
        q.get()

        time_diff = time.time()

        ###train the model
        models[i].fit(train_images, train_labels, epochs=num_of_epochs)
        ###

        time_diff = time.time() - time_diff
        switch(num)
        core_usage = q.get()

        ##get all information needed from model
        test_loss, test_acc = models[i].evaluate(test_images, test_labels)

        info = []
        info.append(test_loss)
        info.append(test_acc)
        info.append(len(hold_core))
        info.append(time_diff)
        info.append(num_of_epochs)
        info.append(i + 1)
        info.append(neuron_per_layer)

        if i == 0:
            title = "Test3"
        else:
            title = ""

        outputAppend(info, core_usage, title)

# creating models that change the number of hidden layers only
def test4(num, q):
    # set variables
    neuron_per_layer = 128

    epoch = [1, 5, 10, 20]

    models.clear()

    ### build the models
    for i in range(len(epoch)):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(neuron_per_layer, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        models.append(model)
    ###

    hold_core = []
    for x in range(cores):
        hold_core.append(x)

    ps.cpu_affinity(hold_core)


    ###
    for i in range(len(models)):
        switch(num)
        q.get()

        time_diff = time.time()

        ###train the model
        models[i].fit(train_images, train_labels, epochs=epoch[i])
        ###

        time_diff = time.time() - time_diff
        switch(num)
        core_usage = q.get()

        ##get all information needed from model
        test_loss, test_acc = models[i].evaluate(test_images, test_labels)

        info = []
        info.append(test_loss)
        info.append(test_acc)
        info.append(len(hold_core))
        info.append(time_diff)
        info.append(epoch[i])
        info.append(1)
        info.append(neuron_per_layer)

        if i == 0:
            title = "Test4"
        else:
            title = ""

        outputAppend(info, core_usage, title)

#add to output to .txt file
def outputAppend(info, core_usage, title):
    if title != "":
        output_file.write("##" + title + "\n")
    output_file.write("Number of cores used:" + str(info[2]) + "\n")
    for i in range(len(core_usage)):
        output_file.write("Core [" + str(i+1) + "]: " + str(core_usage[i]) + "\n")
    output_file.write("Number of neurons per layer used:" + str(info[6]) + "\n")
    output_file.write("Number of hidden layers used:" + str(info[5]) + "\n")
    output_file.write("Number of epochs used:" + str(info[4]) + "\n")
    output_file.write("Loss of model:" + str(info[0]) + "\n")
    output_file.write("Accuracy of model:" + str(info[1]) + "\n")
    output_file.write("Time to train model (s):" + str(info[3]) + "\n")
    output_file.write("\n\n")


###


#preproccessing

#change the number of hidden layers (1, 4, 8)
hidden = [1, 4, 8]
hidden_layers = hidden[0]
#change the number of neurons per layer (32, 128, 256)
neuron = [32, 128, 256, 512]
neuron_per_layer = neuron[1]
#change the number of epochs (1, 10, 20)
epoch = [1, 5, 10, 20]
num_of_epochs = epoch[1]


cores = psutil.cpu_count()
ps = psutil.Process()


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

models = []





##main script

if __name__ == '__main__':
    output_file = open("output_results.txt", "w")

    lock = Lock()
    q = Queue()
    num = Value('i', 0)
    p = Process(target=f, args=(q, num, lock))
    print("start process")
    p.start()
    test2(num, q)
    test3(num, q)
    test4(num, q)
    #test1(num, q)
    num.value = 3
    p.join()

    output_file.close()




