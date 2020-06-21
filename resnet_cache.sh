mpiexec -n=4 python compress_model_parallel.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_parallel_4_gpu_fixed_calls/" \
						-tp="resnet_four_gpu_cifar10_fixed_calls.json" \
						-ep=30

python compress_model.py \
			-bs=256 \
			-im=64 \
			-sp=".summarys/resnet/cifar10_fixed_calls/" \
			-tp="resnet_cifar10_fixed_calls.json" \
			-ep=30

