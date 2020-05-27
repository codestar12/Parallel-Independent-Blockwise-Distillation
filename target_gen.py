
import os
import tensorflow as tf
import json



IMAGE_SIZE = (224, 224)
TRAIN_SIZE = 50000
VALIDATION_SIZE = 10000
BATCH_SIZE_PER_GPU = 64
global_batch_size = (BATCH_SIZE_PER_GPU * 1)
NUM_CLASSES = 10
TEST = 100
EPOCHS = 64 if TEST == 1 else 2
NUM_PROC = 2


model = tf.keras.models.load_model('./base_model_cifar10_vgg16.h5')
model.compile(optimizer=tf.optimizers.SGD(learning_rate=.01, momentum=.9, nesterov=True), loss='mse', metrics=['acc'])

import pprint
targets = []
for i, layer in enumerate(model.layers):
    if layer.__class__.__name__ == "Conv2D":
        if layer.kernel_size[0] == 3:
            #print(f'{i} layer {layer.name} , kernel size {layer.kernel_size}')
            targets.append({'name': layer.name, 'layer': i})


with open('targets.json', 'w') as fp:
	json.dump(targets, fp)