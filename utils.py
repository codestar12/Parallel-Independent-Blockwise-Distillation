import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from typing import Tuple
import math

temp = tf.zeros([4, 32, 32, 3])  # Or tf.zeros
preprocess_input(temp)

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

def zoom(x: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% cropa
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img, image_size):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=image_size)
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(())

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x, image_size))

def normalize(input_image):
  return preprocess_input(input_image)

@tf.function
def load_image_train(datapoint, image_size: Tuple[int, int], num_classes: int):
  input_image, label = tf.image.resize(datapoint["image"], image_size), datapoint['label']
  # if tf.random.uniform(()) > 0.5:
  #   input_image = tf.image.flip_left_right(input_image)
  augmentations = [flip, color, rotate]
  for f in augmentations:
    input_image = tf.cond(tf.random.uniform(()) > 0.75, lambda: f(input_image), lambda: input_image)

  input_image = tf.cond(tf.random.uniform(()) > 0.75, lambda: zoom(input_image, image_size), lambda: input_image)

  #input_image = preprocess_input(input_image)
  input_image = normalize(input_image)

  return input_image, tf.one_hot(label, depth=num_classes)

@tf.function
def load_image_test(datapoint, image_size: Tuple[int, int], num_classes: int):
  input_image, label = tf.image.resize(datapoint["image"], image_size), datapoint['label']
  #input_image = preprocess_input(input_image)

  input_image = normalize(input_image)

  return input_image, tf.one_hot(label, depth=num_classes)

class LayerBatch(tf.keras.utils.Sequence):

	def __init__(self, input_model, dataset, data_size: int, batch_size: int):
		self.input_model = input_model
		self.dataset = dataset.__iter__()
		self.data_size = data_size
		self.batch_size = batch_size

	def __len__(self):
		return math.ceil(self.data_size / self.batch_size)

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



