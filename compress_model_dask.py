
# from dask_cuda import LocalCUDACluster
# from dask.distributed import Client
# from dask.distributed import as_completed
# import dask


# import numpy as np
# import math
# import time

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.distributed import as_completed
from dask.distributed import get_worker
import dask


if __name__ == '__main__':
	import json
	import functools
	import operator
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
	parser.add_argument("-ar", "--arch", type=str,
						help="model architecture being compressed ex. vgg, resnet",
						choices=['vgg', 'resnet'], default='resnet')
	parser.add_argument("-mp", "--model_path", type=str, help="file path to saved model file", default='cifar10.h5')
	parser.add_argument('-aug', "--augment_data", type=bool, default=True, help="Whether or not to augement images or cache them")
	parser.add_argument('-ds', "--dataset", type=str, choices=['cifar10', 'imagenet'], default="cifar10")

	args = parser.parse_args()

	from dask.distributed import SSHCluster
	#cluster = LocalCUDACluster()
	#cluster = SSHCluster(
	#	['localhost', 'localhost', 'localhost', 'localhost', 'localhost'],
	#	connect_options={'known_hosts': None},:
	#	scheduler_options={'port': 0, "dashboard_address": ":8787"},
	#	worker_module='dask_cuda.dask_cuda_worker'
	#)
	client = Client('tcp://127.0.0.1:8786')
	from blockwise.train_layer import train_layer, get_targets, evaluate_model





	score = dask.delayed(evaluate_model)(args)
	score = dask.compute(score)

	print(score)

	targets = get_targets(args)

	def train_dask(target, args):
		print(get_worker().name)
		return train_layer(target, args, get_worker().name)

	targets = [dask.delayed(train_dask)(target, args) for target in targets]

	targets = dask.compute(*targets)
	#targets = client.map(lambda x: train_layer(x, args, targets)
	final_fine_tune = final_fine_tune(targets, args, score)

	if timing_path is not None:
		timing_dump = [{'name': target['name'], 'layer': target['layer'], 'run_time': target['run_time'], 'rank': target['rank'], 'replaced': target['replaced'], 'score': target['score']} for target in targets]
		timing_dump.append({'total_time': total_time})
		timing_dump.append({'final_acc': final[1]})
		timing_dump.append({'fine_tune_acc': final_fine_tune[1]})
		with open(timing_path, 'w') as f:
			json.dump(timing_dump, f, indent='\t')


