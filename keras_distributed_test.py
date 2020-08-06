from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.distributed import as_completed
from dask.distributed import get_worker
import dask
import time

@dask.delayed
def dask_test(split: int, epochs: int):
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras import models
    import tensorflow_datasets as tfds
    from tensorflow.keras.applications.resnet import preprocess_input
    import numpy as np
    import math
    import time
    from typing import Tuple

    temp = tf.zeros([4, 32, 32, 3])
    preprocess_input(temp)

    image_size = (64, 64)


    
    @tf.function
    def load_image(datapoint, image_size: Tuple[int, int], num_classes: int):
        input_image, label = tf.image.resize(datapoint["image"], image_size), datapoint['label']

        input_image = preprocess_input(input_image)

        return input_image, tf.one_hot(label, depth=num_classes)

    model = models.load_model("/tf/notebooks/cifar10.h5")

    dataset = tfds.load('cifar10', split=f'test[:{split}%]')
    dataset = dataset.map(lambda x: load_image(x, image_size, 10))
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model.compile(optimizer="rmsprop", 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    tik = time.time()

    for epoch in range(epochs):
        og = model.evaluate(dataset, verbose=0)

    tok = time.time()

    return og, tok - tik, get_worker().name



if __name__ == '__main__':

    import random
    import time

    cluster = LocalCUDACluster()
    client = Client(cluster) 

    tik = time.time()
    train_sess = [(30, 1), (90, 10), (10, 1), (15, 2), (10, 1), (85, 5), (85, 9), (10, 1), (10, 1), (10, 1), (85, 5), (90, 3), (20, 2), (20, 4)]
    splits = [dask_test(*sess) for sess in train_sess]
    splits = dask.compute(*splits)

    tok = time.time()

    print(splits)

    print(f'total time {tok - tik}')

