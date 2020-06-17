mpiexec -n=4 python compress_model_parallel.py \
						-bs=32 \
						-im=224 \
						-sp=".summarys/vgg/cifar10_parallel_4_gpu/" \
						-tp="./timing_info/vgg/four_gpu_cifar10_224.json" \
						-ep=2 \
						-mp="./base_model_cifar10_vgg16.h5" \
						-tm=1 \
						-ar="vgg" \
						-sd="layer_schedules/vgg16/4gpu_opt_last.json" \
						-tf="targets.json"


python compress_model.py \
			-bs=32 \
			-im=224 \
			-sp=".summarys/vgg/cifar10_224/" \
			-tp="./timing_info/vgg/cifar10_224.json" \
			-ep=2 \
			-mp="./base_model_cifar10_vgg16.h5" \
			-tm=1 \
			-ar="vgg"
