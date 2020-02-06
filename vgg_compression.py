#%%


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#%%
import tensorflow as tf

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import math


#used to fix bug in keras preprocessing scope
temp = tf.zeros([4, 32, 32, 3])  # Or tf.zeros
preprocess_input(temp)
print("processed")


#%%
IMAGE_SIZE = (224, 224)
TRAIN_SIZE = 50000
VALIDATION_SIZE = 10000
BATCH_SIZE_PER_GPU = 16
global_batch_size = (BATCH_SIZE_PER_GPU * 1)
NUM_CLASSES = 10
EPOCHS = 64
TEST = 1

#%% [markdown]
# Dataset code

#%%
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=IMAGE_SIZE)
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(())

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def normalize(input_image):
  return preprocess_input(input_image)

@tf.function
def load_image_train(datapoint):
  input_image, label = tf.image.resize(datapoint["image"], IMAGE_SIZE), datapoint['label']
  # if tf.random.uniform(()) > 0.5:
  #   input_image = tf.image.flip_left_right(input_image)
  augmentations = [flip, color, zoom, rotate]
  for f in augmentations:
    input_image = tf.cond(tf.random.uniform(()) > 0.75, lambda: f(input_image), lambda: input_image)

  #input_image = preprocess_input(input_image)
  input_image = normalize(input_image)

  return input_image, tf.one_hot(label, depth=NUM_CLASSES)

@tf.function
def load_image_test(datapoint):
  input_image, label = tf.image.resize(datapoint["image"], IMAGE_SIZE), datapoint['label']
  #input_image = preprocess_input(input_image)

  input_image = normalize(input_image)

  return input_image, tf.one_hot(label, depth=NUM_CLASSES)

class LayerBatch(tf.keras.utils.Sequence):
    
    def __init__(self, input_model, dataset):
        self.input_model = input_model
        self.dataset = dataset.__iter__()
        
    def __len__(self):
        return math.ceil(TRAIN_SIZE // global_batch_size )
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y
    
class LayerBatchSynth(tf.keras.utils.Sequence):
    
    def __init__(self, input_model, dataset):
        self.input_model = input_model
        self.dataset = dataset.__iter__()
        
    def __len__(self):
        return math.ceil(4224 // global_batch_size )
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y
    
import math
class LayerTest(tf.keras.utils.Sequence):
    
    def __init__(self, input_model, dataset):
        self.input_model = input_model
        self.dataset = dataset.__iter__()
        
    def __len__(self):
        return math.ceil(VALIDATION_SIZE // global_batch_size )
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y

def add_layers(inputs, filters, layers=2):
    print(inputs.get_shape())
    X = tf.keras.layers.SeparableConv2D(name=f'sep_conv_{build_replacement.counter}', filters=filters, 
                                        kernel_size= (3,3),
                                        padding='Same')(inputs)
    #X = tf.keras.layers.BatchNormalization(name=f'batch_norm_{build_replacement.counter}')(X)
    X = tf.keras.layers.ReLU(name=f'relu_{build_replacement.counter}')(X)
    
    build_replacement.counter += 1
    
    for i in range(1, layers):
        X = tf.keras.layers.SeparableConv2D(name=f'sep_conv_{build_replacement.counter}', filters=filters,
                                            kernel_size=(3,3), 
                                            padding='Same')(X)
        #X = tf.keras.layers.BatchNormalization(name=f'batch_norm_{build_replacement.counter}')(X)
        X = tf.keras.layers.ReLU(name=f'relu_{build_replacement.counter}')(X)
        build_replacement.counter += 1
    
    return X
    
def build_replacement(get_output, layers=2):
    inputs = tf.keras.Input(shape=get_output.output[0].shape[1::])
    
    X = add_layers(inputs, get_output.output[1].shape[-1], layers)
    replacement_layers = tf.keras.Model(inputs=inputs, outputs=X)
    return replacement_layers

build_replacement.counter = 0

def replac(inp, filters):
    
    return add_layers(inp, filters,layers=2)

def make_list(X):
    if isinstance(X, list):
        return X
    return [X]

def list_no_list(X):
    if len(X) == 1:
        return X[0]
    return X

def replace_layer(model, replace_layer_subname, replacement_fn,
**kwargs):
    """
    args:
        model :: keras.models.Model instance
        replace_layer_subname :: str -- if str in layer name, replace it
        replacement_fn :: fn to call to replace all instances
            > fn output must produce shape as the replaced layers input
    returns:
        new model with replaced layers
    quick examples:
        want to just remove all layers with 'batch_norm' in the name:
            > new_model = replace_layer(model, 'batch_norm', lambda **kwargs : (lambda u:u))
        want to replace all Conv1D(N, m, padding='same') with an LSTM (lets say all have 'conv1d' in name)
            > new_model = replace_layer(model, 'conv1d', lambda layer, **kwargs: LSTM(units=layer.filters, return_sequences=True)
    """
    model_inputs = []
    model_outputs = []
    tsr_dict = {}

    model_output_names = [out.name for out in make_list(model.output)]

    for i, layer in enumerate(model.layers):
        ### Loop if layer is used multiple times
        for j in range(len(layer._inbound_nodes)):

            ### check layer inp/outp
            inpt_names = [inp.name for inp in make_list(layer.get_input_at(j))]
            outp_names = [out.name for out in make_list(layer.get_output_at(j))]

            ### setup model inputs
            if 'input' in layer.name:
                for inpt_tsr in make_list(layer.get_output_at(j)):
                    model_inputs.append(inpt_tsr)
                    tsr_dict[inpt_tsr.name] = inpt_tsr
                continue

            ### setup layer inputs
            # I added the exception model_3_3/Identity:0 I think the problem is that is the input layer
            inpt = list_no_list([tsr_dict[name]  for name in inpt_names])

            ### remake layer 
            if layer.name in replace_layer_subname:
              if "relu" in layer.name or 'bn' in layer.name:
                print('deleting ' + layer.name)
                x = inpt
              else:
                print('replacing '+layer.name)
                x = replacement_fn(inpt)
            else:
                x = layer(inpt)

            ### reinstantialize outputs into dict
            for name, out_tsr in zip(outp_names, make_list(x)):

                ### check if is an output
                if name in model_output_names:
                    model_outputs.append(out_tsr)
                tsr_dict[name] = out_tsr

    return tf.keras.models.Model(model_inputs, model_outputs)

dataset, info = tfds.load('cifar10', with_info=True)



train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train.shuffle(buffer_size=1000).batch(global_batch_size).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


test_dataset = dataset['test'].map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(global_batch_size).repeat()
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


model = tf.keras.models.load_model('./base_model_cifar10_vgg16.h5')
model.compile(optimizer=tf.optimizers.SGD(learning_rate=.01, momentum=.9, nesterov=True), loss='mse', metrics=['acc'])
OG = model.evaluate(test_dataset, steps=VALIDATION_SIZE//global_batch_size//TEST)
print(OG)


#%%



#%%
import pprint
targets = []
for i, layer in enumerate(model.layers):
    if layer.__class__.__name__ == "Conv2D":
        if layer.kernel_size[0] == 3:
            #print(f'{i} layer {layer.name} , kernel size {layer.kernel_size}')
            targets.append({'name': layer.name, 'layer': i})

pprint.pprint(targets)



for target in targets:

    writer = tf.summary.create_file_writer(f"./summarys/{target['name']}")
    with writer.as_default():
        print(f"training layer {target['name']}")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model('base_model_cifar10_vgg16.h5')
        in_layer = target['layer']
        get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output, 
                                                                model.layers[in_layer].output])


        replacement_layers = build_replacement(get_output, layers=2)
        replacement_len = len(replacement_layers.layers)
        layer_train_gen = LayerBatch(get_output, train_dataset)
        layer_test_gen = LayerTest(get_output, test_dataset)




        MSE = tf.losses.MeanSquaredError()

        optimizer=tf.keras.optimizers.SGD(.00001, momentum=.9, nesterov=True)
        replacement_layers.compile(loss=MSE, optimizer=optimizer)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=.0001, factor=.3, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=15, min_delta=.0001, restore_best_weights=True, verbose=1)

        replacement_layers.save('/tmp/layer.h5')

        for epoch in range(EPOCHS + 1):

            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model('base_model_cifar10_vgg16.h5')
            in_layer = target['layer']
            get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output, 
                                                                    model.layers[in_layer].output])


            
            layer_train_gen = LayerBatch(get_output, train_dataset)
            layer_test_gen = LayerTest(get_output, test_dataset)

            replacement_layers = tf.keras.models.load_model('/tmp/layer.h5')

            history = replacement_layers.fit(x=layer_train_gen,
                                        epochs=1,
                                        steps_per_epoch=TRAIN_SIZE // global_batch_size // TEST,
                                        validation_data=layer_test_gen,
                                        shuffle=False,
                                        callbacks=[reduce_lr, early_stop],
                                        validation_steps=VALIDATION_SIZE // global_batch_size // TEST,
                                        verbose=0)
            
            replacement_layers.save('/tmp/layer.h5')

            target['weights'] = [replacement_layers.layers[1].get_weights(), replacement_layers.layers[3].get_weights()]

            tf.keras.backend.clear_session()

            model = tf.keras.models.load_model('base_model_cifar10_vgg16.h5')
            layer_name = target['name']
            layer_pos = target['layer']
            filters = model.layers[layer_pos].output.shape[-1]

            new_model = replace_layer(model, layer_name, lambda x: replac(x, filters))
            new_model.layers[layer_pos].set_weights(target['weights'][0])
            new_model.layers[layer_pos + 2].set_weights(target['weights'][1])
            new_model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
            target['score'] = new_model.evaluate(test_dataset, steps=VALIDATION_SIZE // global_batch_size // TEST)

            tf.summary.scalar(name='rep_loss', data=history.history['loss'][0], step=epoch)
            tf.summary.scalar(name='val_loss', data=history.history['val_loss'][0], step=epoch)
            tf.summary.scalar(name='model_acc', data=target['score'][1], step=epoch)
            tf.summary.scalar(name='model_loss', data=target['score'][0], step=epoch)

            writer.flush()
            print(f"epoch: {epoch}, rep loss {history.history['loss']}, val loss {history.history['val_loss']}, model acc {target['score'][1]}")

            if np.abs(OG[1] - target['score'][1]) < 0.0001:
                print('stoping early')
                break
    


        





#%%
# for target in targets:
    
#     print(f'Replacing Layer {target["name"]}')
    
#     tf.keras.backend.clear_session()
    
#     model = tf.keras.models.load_model('base_model_cifar10_vgg16.h5')
    
#     layer_name = target['name']
#     layer_pos = target['layer']
#     filters = model.layers[layer_pos].output.shape[-1]
    
    
#     new_model = replace_layer(model, layer_name, lambda x: replac(x, filters))
#     new_model.layers[layer_pos].set_weights(target['weights'][0])
#     new_model.layers[layer_pos + 2].set_weights(target['weights'][1])
#     new_model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
#     target['score'] = new_model.evaluate(test_dataset, steps=VALIDATION_SIZE // global_batch_size)
    



