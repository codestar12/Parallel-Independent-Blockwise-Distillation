

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import math
from mpi4py import MPI
import tensorflow as tf
IMAGE_SIZE = (224, 224)
TRAIN_SIZE = 50000
VALIDATION_SIZE = 10000
BATCH_SIZE_PER_GPU = 64
global_batch_size = (BATCH_SIZE_PER_GPU * 1)
NUM_CLASSES = 10
TEST = 100
EPOCHS = 64 if TEST == 1 else 2
NUM_PROC = 2

def train_layer(target):

	"""Trains a replacement layer given a target

	Args:
		target: A dictonary containing {'name': layer.name, 'layer': i}

	Returns:
		target: Updated dictonary
	"""

	import tensorflow as tf
	import tensorflow_datasets as tfds
	from utils import load_image_train, load_image_test, build_replacement, LayerBatch, LayerTest, replac, replace_layer

	dataset, info = tfds.load('cifar10', with_info=True)

	train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	train_dataset = train.shuffle(buffer_size=1000).batch(global_batch_size).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	test_dataset = dataset['test'].map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_dataset = test_dataset.batch(global_batch_size).repeat()
	test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	writer = tf.summary.create_file_writer(f"./summarys/vgg/cifar10_parallel/{target['name']}")
	with writer.as_default():
		print(f"training layer {target['name']}")
		tf.keras.backend.clear_session()
		print("cleared backend")
		model = tf.keras.models.load_model('base_model_cifar10_vgg16.h5')
		print("model loaded")
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

		replacement_layers.save(f'/tmp/layer_{pid}.h5')

		print('epochs started')
		for epoch in range(EPOCHS + 1):

			tf.keras.backend.clear_session()
			model = tf.keras.models.load_model('base_model_cifar10_vgg16.h5')
			in_layer = target['layer']
			get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output,
																	model.layers[in_layer].output])



			layer_train_gen = LayerBatch(get_output, train_dataset)
			layer_test_gen = LayerTest(get_output, test_dataset)

			replacement_layers = tf.keras.models.load_model(f'/tmp/layer_{pid}.h5')

			print('training started')
			history = replacement_layers.fit(x=layer_train_gen,
										epochs=1,
										steps_per_epoch=TRAIN_SIZE // global_batch_size // TEST,
										validation_data=layer_test_gen,
										shuffle=False,
										callbacks=[reduce_lr, early_stop],
										validation_steps=VALIDATION_SIZE // global_batch_size // TEST,
										verbose=1)

			replacement_layers.save(f'/tmp/layer_{pid}.h5')

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

	return target


if __name__ == '__main__':
	import json 

	with open('targets.json', 'r') as f:
		targets = json.load(f)

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()	

	tf.debugging
	with tf.device(f'/GPU:{rank}'):
		a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
		b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

		c = tf.matmul(a, b)

	if rank == 0:
		print(c)
	

	#print(targets)
	# tf.keras.backend.clear_session()
	# model = tf.keras.models.load_model('./base_model_cifar10_vgg16.h5')


	# writer = tf.summary.create_file_writer(f"./summarys/vgg/cifar10_parallel/final_model")
	# with writer.as_default():
	# 	for target in targets[::-1]:
	# 		print(f'replacing layer {target["name"]}')

	# 		layer_name = target['name']
	# 		layer_pos = target['layer']
	# 		filters = model.layers[layer_pos].output.shape[-1]



	# 		new_model = replace_layer(model, layer_name, lambda x: replac(x, filters))
	# 		new_model.layers[layer_pos].set_weights(target['weights'][0])
	# 		new_model.layers[layer_pos + 2].set_weights(target['weights'][1])

	# 		new_model.save('cifar10_vgg_modified.h5')
	# 		tf.keras.backend.clear_session()
	# 		model = tf.keras.models.load_model('cifar10_vgg_modified.h5')

	# 	tf.keras.backend.clear_session()
	# 	model = tf.keras.models.load_model('cifar10_vgg_modified.h5')
	# 	model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
	# 	final = model.evaluate(test_dataset, steps=VALIDATION_SIZE // global_batch_size // TEST)

	# 	tf.summary.scalar(name='model_acc', data=final[1], step=0)
	# 	tf.summary.scalar(name='model_loss', data=final[0], step=0)

	# 	writer.flush()
