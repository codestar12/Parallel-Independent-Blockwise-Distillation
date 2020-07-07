mpiexec -n=4 python compress_model_parallel_imagenet.py \
						-bs=32 \
						-im=224 \
						-sp=".summarys/resnet/imagenet_parallel_4_gpu_fine_tune/" \
						-tp="resnet_four_gpu_imagenet_fine_tune.json" \
						-ep=30

mpiexec -n=4 python compress_model_parallel_imagenet.py \
						-bs=32 \
						-im=224 \
						-sp=".summarys/resnet/imagenet_parallel_4_gpu_fine_tune_freeze/" \
						-tp="resnet_four_gpu_imagenet_fine_tune_freeze.json" \
						-ep=30 \
						-fr=True

 #mpiexec -n=3 python compress_model_parallel_imagenet.py \
 #						-bs=32 \
 #						-im=224 \
 #						-sp=".summarys/resnet/cifar10_parallel_3_gpu_fixed_calls/" \
 #						-tp="resnet_three_gpu_cifar10_fixed_calls.json" \
 #						-ep=30

# mpiexec -n=2 python compress_model_parallel.py \
# 						-bs=256 \
# 						-im=64 \
# 						-sp=".summarys/resnet/cifar10_parallel_2_gpu_fixed_calls/" \
# 						-tp="resnet_two_gpu_cifar10_fixed_calls.json" \
# 						-ep=30

# python compress_model.py \
# 			-bs=256 \
# 			-im=64 \
# 			-sp=".summarys/resnet/cifar10_fixed_calls/" \
# 			-tp="resnet_cifar10_fixed_calls.json" \
# 			-ep=30

# mpiexec -n=4 python compress_model_parallel.py \
# 						-bs=256 \
# 						-im=64 \
# 						-sp=".summarys/resnet/cifar10_parallel_4_gpu_fixed_calls_cache/" \
# 						-tp="resnet_four_gpu_cifar10_fixed_calls_cache.json" \
# 						-ep=30 \
# 						-aug=False

# python compress_model.py \
# 			-bs=256 \
# 			-im=64 \
# 			-sp=".summarys/resnet/cifar10_fixed_calls_cache/" \
# 			-tp="resnet_cifar10_fixed_calls_cache.json" \
# 			-ep=30 \
# 			-aug=False
