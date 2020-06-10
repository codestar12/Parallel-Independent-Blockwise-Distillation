IMAGE_SIZE = (64, 64)
TRAIN_SIZE = 50000
VALIDATION_SIZE = 10000
BATCH_SIZE_PER_GPU = 512
global_batch_size = (BATCH_SIZE_PER_GPU * 1)
NUM_CLASSES = 10
TEST = 1
EPOCHS = 20 if TEST == 1 else 2
NUM_PROC = 2
EARLY_STOPPING = False
SUMMARY_PATH = ""
OG = None
ARCH = 'resnet'
MODEL_PATH = ""
TARGET_FILE = ""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
import tensorflow_datasets as tfds
if ARCH == 'resnet': 
	from utils_resnet import * 
elif ARCH == 'vgg':
	from utils import *
import numpy as np
import math
import time


def train_layer(target, rank=0):

	"""Trains a replacement layer given a target

	Args:
		target: A dictionary containing {'name': layer.name, 'layer': i}

	Returns:
		target: Updated dictionary
	"""

	layer_start = time.time()
	dataset, info = tfds.load('cifar10', with_info=True)

	train = dataset['train'].map(lambda x: load_image_train(x, IMAGE_SIZE, NUM_CLASSES), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	train_dataset = train.shuffle(buffer_size=TRAIN_SIZE).batch(global_batch_size).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	test_dataset = dataset['test'].map(lambda x: load_image_test(x, IMAGE_SIZE, NUM_CLASSES), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_dataset = test_dataset.batch(global_batch_size).repeat()
	test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	writer = tf.summary.create_file_writer(SUMMARY_PATH + f"{target['name']}")
	with writer.as_default():
		print(f"training layer {target['name']}")
		tf.keras.backend.clear_session()
		print("cleared backend")
		model = tf.keras.models.load_model(MODEL_PATH)
		print("model loaded")
		in_layer = target['layer']
		get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output,
																model.layers[in_layer].output])


		replacement_layers = build_replacement(get_output, layers=2)
		replacement_len = len(replacement_layers.layers)
		layer_train_gen = LayerBatch(get_output, train_dataset, TRAIN_SIZE, global_batch_size)
		layer_test_gen = LayerBatch(get_output, test_dataset, VALIDATION_SIZE, global_batch_size)




		MSE = tf.losses.MeanSquaredError()

		starting_lr = 2e-2
		initial_learning_rate = 0.1
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			starting_lr,
			decay_steps=3000,
			decay_rate=0.96,
			staircase=True)

		optimizer=tf.keras.optimizers.RMSprop(lr_schedule)
		replacement_layers.compile(loss=MSE, optimizer=optimizer)


		replacement_layers.save(f'/tmp/layer_{rank}.h5')

		print('epochs started')
		for epoch in range(EPOCHS):

			if epoch % 2 == 0:
				tf.keras.backend.clear_session()
				model = tf.keras.models.load_model(MODEL_PATH)
				in_layer = target['layer']
				get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output,
																		model.layers[in_layer].output])



				layer_train_gen = LayerBatch(get_output, train_dataset, TRAIN_SIZE, global_batch_size)
				layer_test_gen = LayerBatch(get_output, test_dataset, VALIDATION_SIZE, global_batch_size)

				replacement_layers = tf.keras.models.load_model(f'/tmp/layer_{rank}.h5')

			print('training started')
			history = replacement_layers.fit(x=layer_train_gen,
										epochs=1,
										steps_per_epoch=math.ceil(TRAIN_SIZE / global_batch_size / TEST),
										validation_data=layer_test_gen,
										shuffle=False,
										validation_steps=math.ceil(VALIDATION_SIZE / global_batch_size / TEST),
										verbose=2)

			tf.summary.scalar(name='rep_loss', data=history.history['loss'][0], step=epoch)
			tf.summary.scalar(name='val_loss', data=history.history['val_loss'][0], step=epoch)

			if epoch % 2 == 0:
				replacement_layers.save(f'/tmp/layer_{rank}.h5')

				target['weights'] = [replacement_layers.layers[1].get_weights(), replacement_layers.layers[3].get_weights()]

				tf.keras.backend.clear_session()

				model = tf.keras.models.load_model(MODEL_PATH)
				layer_name = target['name']
				layer_pos = target['layer']
				filters = model.layers[layer_pos].output.shape[-1]

				new_model = replace_layer(model, layer_name, lambda x: replac(x, filters))
				new_model.layers[layer_pos].set_weights(target['weights'][0])
				new_model.layers[layer_pos + 2].set_weights(target['weights'][1])
				new_model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
				target['score'] = new_model.evaluate(test_dataset, steps=math.ceil(VALIDATION_SIZE / global_batch_size / TEST))

				tf.summary.scalar(name='model_acc', data=target['score'][1], step=epoch)
				tf.summary.scalar(name='model_loss', data=target['score'][0], step=epoch)

				print(f"epoch: {epoch}, rep loss {history.history['loss']}, val loss {history.history['val_loss']}, model acc {target['score'][1]}")

				if EARLY_STOPPING:
					if OG[1] - target['score'][1] < 0.002:
						print('stoping early')
						break

			writer.flush()

	layer_end = time.time()
	layer_time = layer_end - layer_start
	target['run_time'] = layer_time
	target['rank'] = rank
	return target


if __name__ == '__main__':
	import json
	import functools
	import operator
	import tensorflow_datasets as tfds
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-im", "--image_size", type=int,
						help="dataset image size", default=32)
	parser.add_argument("-ts", "--train_size", type=int,
						help="dataset training split size", default=50000)
	parser.add_argument("-vs", "--val_size", type=int,
						help="dataset validation split size", default=10000)
	parser.add_argument("-bs", "--batch_size", type=int,
						help="batch size", default=256)
	parser.add_argument("-nc", "--num_classes", type=int, default=10)
	parser.add_argument("-ep", "--epochs", type=int, default=40)
	parser.add_argument("-es", "--early_stopping", type=bool, default=False)
	parser.add_argument("-tm", "--test_multiplier", type=int, default=1,
						help="multipler to speed up training when testing")
	parser.add_argument("-sp", "--summary_path", type=str, default="./summarys/vgg/")
	parser.add_argument("-tp", "--timing_path", type=str, help="file name and path for saving timing data")
	parser.add_argument("-sd", "--schedule", type=str, help="file name and path for schedule")
	parser.add_argument("-ar", "--arch", type=str, 
						help="model architecture being compressed ex. vgg, resnet",
						choices=['vgg', 'resnet'], default='resnet')
	parser.add_argument("-mp", "--model_path", type=str, help="file path to saved model file", default='cifar10.h5')
	parser.add_argument("-tp", "target_file_path", type=str, help="path to target file", default='targets_resnet.json')

	args = parser.parse_args()
	IMAGE_SIZE = (args.image_size, args.image_size)
	TRAIN_SIZE = args.train_size
	VALIDATION_SIZE = args.val_size
	global_batch_size = args.batch_size
	NUM_CLASSES = args.num_classes
	EPOCHS = args.epochs
	EARLY_STOPPING = args.early_stopping
	TEST = args.test_multiplier
	SUMMARY_PATH = args.summary_path
	timing_path = args.timing_path
	schedule = args.schedule
	ARCH = args.arch
	MODEL_PATH = args.model_path
	TARGET_FILE = args.target_file_path

	with open(TARGET_FILE, 'r') as f:
		targets = json.load(f)


	dataset, info = tfds.load('cifar10', with_info=True)
	test_dataset = dataset['test'].map(lambda x: load_image_test(x, IMAGE_SIZE, NUM_CLASSES), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_dataset = test_dataset.batch(global_batch_size).repeat()
	test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


	model = tf.keras.models.load_model(MODEL_PATH)
	model.compile(optimizer=tf.optimizers.SGD(learning_rate=.01, momentum=.9, nesterov=True), loss='mse', metrics=['acc'])
	OG = model.evaluate(test_dataset, steps=math.ceil(VALIDATION_SIZE/global_batch_size/TEST))
	del model
	tf.keras.backend.clear_session()

	my_layers = []
	# either load a schedule from json for each GPU or
	# assign one based on cyclic assignment
	if schedule is not None:
		with open(schedule, 'r') as f:
			schedules = json.load(f)
		my_layers = schedules[str(rank)]
	else:
		my_layers = [targets[i]['layer'] for i in range(rank, len(targets), size)]


	if rank == 0:
		tik = time.time()


	print(f"OG IS : {OG}\n")
	print(my_layers)
	with tf.device(f'/GPU:{rank}'):
		print(f'here from {rank}')
		targets = [train_layer(target, rank) for target in targets if target['layer'] in my_layers]


	targets = comm.gather(targets, root=0)

	if rank == 0:

		tok = time.time()
		total_time = tok - tik


		targets = functools.reduce(operator.iconcat, targets, [])

		list.sort(targets, key=lambda target: target['layer'])

		if timing_path is not None:
			timing_dump = [{'name': target['name'], 'layer': target['layer'], 'run_time': target['run_time'], 'rank': target['rank']} for target in targets]
			timing_dump.append({'total_time': total_time})
			with open(timing_path, 'w') as f:
				json.dump(timing_dump, f, indent='\t')

		tf.keras.backend.clear_session()
		model = tf.keras.models.load_model(MODEL_PATH)


		writer = tf.summary.create_file_writer(SUMMARY_PATH +  "final_model")
		with writer.as_default():
			for target in targets[::-1]:
				print(f'replacing layer {target["name"]}')

				layer_name = target['name']
				layer_pos = target['layer']
				filters = model.layers[layer_pos].output.shape[-1]



				new_model = replace_layer(model, layer_name, lambda x: replac(x, filters))
				new_model.layers[layer_pos].set_weights(target['weights'][0])
				new_model.layers[layer_pos + 2].set_weights(target['weights'][1])

				new_model.save('cifar10_resnet_modified.h5')
				tf.keras.backend.clear_session()
				model = tf.keras.models.load_model('cifar10_resnet_modified.h5')

			tf.keras.backend.clear_session()

			test_dataset = dataset['test'].map(lambda x: load_image_test(x, IMAGE_SIZE, NUM_CLASSES), num_parallel_calls=tf.data.experimental.AUTOTUNE)
			test_dataset = test_dataset.batch(global_batch_size).repeat()
			test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

			model = tf.keras.models.load_model('cifar10_resnet_modified.h5')
			model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
			final = model.evaluate(test_dataset, steps=math.ceil(VALIDATION_SIZE / global_batch_size / TEST))

			tf.summary.scalar(name='model_acc', data=final[1], step=0)
			tf.summary.scalar(name='model_loss', data=final[0], step=0)

			writer.flush()
