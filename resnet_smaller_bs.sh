mpiexec -n=4 python compress_model_parallel.py \
						-bs=128 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_parallel_4_gpu_bs_64/" \
						-tp="resnet_four_gpu_cifar10_lrsch_bs_64.json" \
						-ep=30

python compress_model.py \
			-bs=128 \
			-im=64 \
			-sp=".summarys/resnet/cifar10_lrsch_bs_64/" \
			-tp="resnet_cifar10_lrsch_bs_64.json" \
			-ep=30

