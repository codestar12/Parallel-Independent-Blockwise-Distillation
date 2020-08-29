def train_layer(target, args, rank=0):

	"""Trains a replacement layer given a target

	Args:
		target: A dictonary containing {'name': layer.name, 'layer': i}

	Returns:
		target: Updated dictonary
	"""
	import time
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	import math
	import tensorflow as tf
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	for i in range(len(physical_devices)):
		tf.config.experimental.set_memory_growth(physical_devices[i], True)
	import tensorflow_datasets as tfds

	image_size = (args.image_size, args.image_size)

	if args.arch == 'resnet':
		from utils_resnet import load_image_train, load_image_test, build_replacement, LayerBatch, replac, replace_layer
	elif args.arch == 'vgg':
		from utils import load_image_train, load_image_test, build_replacement, LayerBatch, replac, replace_layer

	layer_start = time.time()
	if args.dataset == 'cifar10':
		dataset, info = tfds.load('cifar10', with_info=True)
		options = tf.data.Options()
		options.experimental_threading.max_intra_op_parallelism = 1
		train = dataset['train'].with_options(options)
		test = dataset['test'].with_options(options)
	elif args.dataset == 'imagenet':
		dataset, info = tfds.load('imagenet2012', with_info=True)
		options = tf.data.Options()
		options.experimental_threading.max_intra_op_parallelism = 1
		train = dataset['train'].with_options(options)
		test = dataset['validation'].with_options(options)

	if args.augment_data:
		train = train.map(lambda x: load_image_train(x, image_size, args.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	else:
		train = train.map(lambda x: load_image_test(x, image_size, args.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		train = train.cache()
	train_dataset = train.shuffle(buffer_size=4000).batch(args.batch_size).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	test_dataset = test.map(lambda x: load_image_test(x, image_size, args.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	#test_dataset = test_dataset.cache()
	test_dataset = test_dataset.batch(args.batch_size).repeat()
	test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	writer = tf.summary.create_file_writer(args.summary_path + f"{target['name']}")
	with writer.as_default():
		print(f"training layer {target['name']}")
		tf.keras.backend.clear_session()
		print("cleared backend")

		if args.dataset != 'imagenet':
			model = tf.keras.models.load_model(args.model_path)
		else:
			if args.arch == 'resnet':
				model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
			else:
				model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')

		print("model loaded")
		in_layer = target['layer']
		get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output,
																model.layers[in_layer].output])


		replacement_layers = build_replacement(get_output, layers=2)
		replacement_len = len(replacement_layers.layers)
		layer_train_gen = LayerBatch(get_output, train_dataset, args.train_size, args.batch_size)
		layer_test_gen = LayerBatch(get_output, test_dataset, args.val_size, args.batch_size)




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

		target['score'] = (0, 0)
		replacement_layers.save(f'/tmp/layer_{rank}.h5')

		print('epochs started')
		for epoch in range(args.epochs):
			if epoch % 2 == 0:
				tf.keras.backend.clear_session()

				if args.dataset != 'imagenet':
					model = tf.keras.models.load_model(args.model_path)
				else:
					if args.arch == 'resnet':
						model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
					else:
						model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')

				in_layer = target['layer']
				get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[in_layer - 1].output,
																		model.layers[in_layer].output])



				layer_train_gen = LayerBatch(get_output, train_dataset, args.train_size, args.batch_size)
				layer_test_gen = LayerBatch(get_output, test_dataset, args.val_size, args.batch_size)

				replacement_layers = tf.keras.models.load_model(f'/tmp/layer_{rank}.h5')

			print('training started')
			history = replacement_layers.fit(x=layer_train_gen,
										epochs=1,
										steps_per_epoch=math.ceil(args.train_size / args.batch_size / args.test_multiplier),
										validation_data=layer_test_gen,
										shuffle=False,
										validation_steps=math.ceil(args.val_size / args.batch_size / args.test_multiplier),
										verbose=2)

			tf.summary.scalar(name='rep_loss', data=history.history['loss'][0], step=epoch)
			tf.summary.scalar(name='val_loss', data=history.history['val_loss'][0], step=epoch)

			if epoch % 2 == 0:
				replacement_layers.save(f'/tmp/layer_{rank}.h5')

				weights = [replacement_layers.layers[1].get_weights(), replacement_layers.layers[3].get_weights()]

				tf.keras.backend.clear_session()

				if args.dataset != 'imagenet':
					model = tf.keras.models.load_model(args.model_path)
				else:
					if args.arch == 'resnet':
						model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
					else:
						model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')

				layer_name = target['name']
				layer_pos = target['layer']
				filters = model.layers[layer_pos].output.shape[-1]

				new_model = replace_layer(model, layer_name, lambda x: replac(x, filters))
				new_model.layers[layer_pos].set_weights(weights[0])
				new_model.layers[layer_pos + 2].set_weights(weights[1])
				new_model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
				score = new_model.evaluate(test_dataset, steps=math.ceil(args.val_size / args.batch_size / args.test_multiplier))

				if score[1] > target['score'][1]:
					target['score'] = score
					target['weights'] = weights

				tf.summary.scalar(name='model_acc', data=score[1], step=epoch)
				tf.summary.scalar(name='model_loss', data=score[0], step=epoch)

				print(f"epoch: {epoch}, rep loss {history.history['loss']}, val loss {history.history['val_loss']}, model acc {score[1]}")

				if args.early_stopping:
					if np.abs(OG[1] - target['score'][1] < 0.002):
						print('stoping early')
						break

			writer.flush()

	layer_end = time.time()
	layer_time = layer_end - layer_start
	target['run_time'] = layer_time
	target['rank'] = rank
	return target

def get_targets(args):
	import tensorflow as tf
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	for i in range(len(physical_devices)):
			tf.config.experimental.set_memory_growth(physical_devices[i], True)

	from tensorflow.keras import models

	
	if args.dataset != 'imagenet':
		model = tf.keras.models.load_model(args.model_path)
	else:
		if args.arch == 'resnet':
			model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
		else:
			model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')


	targets = []
	for i, layer in enumerate(model.layers):
		if layer.__class__.__name__ == "Conv2D":
			if layer.kernel_size[0] == 3:
				#print(f'{i} layer {layer.name} , kernel size {layer.kernel_size}')
				targets.append({'name': layer.name, 'layer': i})

	return targets

def evaluate_model(args):
	
	import tensorflow as tf
	import tensorflow_datasets as tfds
	import math


	val_size = args.val_size

	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	for i in range(len(physical_devices)):
			tf.config.experimental.set_memory_growth(physical_devices[i], True)


	if args.arch == 'resnet':
		from utils_resnet import load_image_train, load_image_test, build_replacement, LayerBatch, replac, replace_layer
	elif args.arch == 'vgg':
		from utils import load_image_train, load_image_test, build_replacement, LayerBatch, replac, replace_layer   

	if args.dataset == 'cifar10':
		dataset, info = tfds.load(args.dataset, with_info=True)
		test = dataset['test']
		train = dataset['train']
	elif args.dataset == 'imagenet':
		dataset, info = tfds.load('imagenet2012', with_info=True)
		print(info)
		test = dataset['validation']
		train = dataset['train']
		val_size = 50000


	train = train.map(
		lambda x: load_image_train(x, (args.image_size, args.image_size), args.num_classes), 
		num_parallel_calls=tf.data.experimental.AUTOTUNE
	)

	train_dataset = train.shuffle(1000).batch(args.batch_size).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	test_dataset = test.map(
		lambda x: load_image_test(x, (args.image_size, args.image_size), args.num_classes), 
		num_parallel_calls=tf.data.experimental.AUTOTUNE
	)

	test_dataset = test_dataset.batch(args.batch_size).repeat()
	test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	if args.dataset != 'imagenet':
		model = tf.keras.models.load_model(args.model_path)
	else:
		if args.arch == 'resnet':
			model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
		else:
			model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')

	model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="categorical_crossentropy", metrics=['accuracy'])
	score = model.evaluate(test_dataset, steps=math.ceil(val_size / args.batch_size / args.test_multiplier))

	return score

def fine_tune_model(targets, args, score):
	import tensorflow as tf
	import tensorflow_datasets as tfds
	import math

	if args.arch == 'resnet':
		from utils_resnet import load_image_train, load_image_test, build_replacement, LayerBatch, replac, replace_layer
	elif args.arch == 'vgg':
		from utils import load_image_train, load_image_test, build_replacement, LayerBatch, replac, replace_layer 
	
	targets = list(targets)
	list.sort(targets, key=lambda target: target['layer'])

	tf.keras.backend.clear_session()

	if args.dataset != 'imagenet':
		model = tf.keras.models.load_model(args.model_path)
	else:
		if args.arch == 'resnet':
			model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
		else:
			model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
	model.save('cifar10_resnet_modified.h5')
	for target in targets[::-1]:
		target['replaced'] = False
		if score[1] - target['score'][1] < 0.15:
			print(f'replacing layer {target["name"]}')
			target['replaced'] = True
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

	val_size = args.val_size

	if args.dataset == 'cifar10':
		dataset, info = tfds.load(args.dataset, with_info=True)
		test = dataset['test']
		train = dataset['train']
	elif args.dataset == 'imagenet':
		dataset, info = tfds.load('imagenet2012', with_info=True)
		print(info)
		test = dataset['validation']
		train = dataset['train']
		val_size = 50000
		
	test_dataset = test.map(lambda x: load_image_test(x, (args.image_size, args.image_size), args.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_dataset = test_dataset.batch(args.batch_size).repeat()
	test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	
	train = train.map(lambda x: load_image_train(x, (args.image_size, args.image_size), args.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)

	train_dataset = train.batch(args.batch_size).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	fine_tune_epochs = 2
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
				.00063,
				decay_steps= math.ceil(args.train_size / args.batch_size  ) * fine_tune_epochs // 3,
				decay_rate=0.96,
				staircase=False)

	from tensorflow.keras.callbacks import ModelCheckpoint

	checkpoint = ModelCheckpoint('cifar10_resnet_modified_fine_tune.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

	model = tf.keras.models.load_model('cifar10_resnet_modified.h5')
	model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=.9, nesterov=True), loss="categorical_crossentropy", metrics=['accuracy'])
	final = model.evaluate(test_dataset, steps=math.ceil(args.val_size / args.batch_size / args.test_multiplier))
	fine_tune = model.fit(
						x=train_dataset,
						epochs=fine_tune_epochs,
						steps_per_epoch=math.ceil(args.train_size / args.batch_size / args.test_multiplier),
						validation_data=test_dataset,
						shuffle=False,
						validation_steps=math.ceil(args.val_size / args.batch_size / args.test_multiplier),
						verbose=1,
						callbacks=[checkpoint])

	model = tf.keras.models.load_model('cifar10_resnet_modified_fine_tune.h5')
	final_fine_tune = model.evaluate(test_dataset, steps=math.ceil(val_size / args.batch_size / args.test_multiplier))

	return final, final_fine_tune, targets