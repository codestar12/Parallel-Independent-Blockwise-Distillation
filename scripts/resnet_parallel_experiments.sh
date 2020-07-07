mpiexec -n=4 python resnet_compression_parallel_32.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_parallel_4_gpu/" \
						-tp="resnet_four_gpu_cifar10_lrsch.json" \
						-ep=30

mpiexec -n=3 python resnet_compression_parallel_32.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_parallel_3_gpu/" \
						-tp="resnet_three_gpu_cifar10_lrsch.json" \
						-ep=30

mpiexec -n=2 python resnet_compression_parallel_32.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_parallel_2_gpu/" \
						-tp="resnet_two_gpu_cifar10_lrsch.json" \
						-ep=30

python resnet_compression_32.py \
			-bs=256 \
			-im=64 \
			-sp=".summarys/resnet/cifar10_lrsch/" \
			-tp="resnet_cifar10_lrsch.json" \
			-ep=30

mpiexec -n=4 python resnet_compression_parallel_32.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_early_stop_parallel_4_gpu/" \
						-tp="resnet_four_gpu_cifar10_lrsch_earlystop.json" \
						-ep=30 \
						-es=True

mpiexec -n=3 python resnet_compression_parallel_32.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_early_stop_parallel_3_gpu/" \
						-tp="resnet_three_gpu_cifar10_lrsch_earlystop.json" \
						-ep=30 \
						-es=True

mpiexec -n=2 python resnet_compression_parallel_32.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_lrsch_early_stop_parallel_2_gpu/" \
						-tp="resnet_two_gpu_cifar10_lrsch_earlystop.json" \
						-ep=30 \
						-es=True

python resnet_compression_32.py \
			-bs=256 \
			-im=64 \
			-sp=".summarys/resnet/cifar10_lrsch_early_stop/" \
			-tp="resnet_cifar10_lrsch_earlystop.json" \
			-ep=30 \
			-es=True
