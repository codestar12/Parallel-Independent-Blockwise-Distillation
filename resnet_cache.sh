# mpiexec -n=4 python compress_model_parallel.py \
# 						-bs=256 \
# 						-im=64 \
# 						-sp=".summarys/resnet/cifar10_4_epochs_high_acc/" \
# 						-tp="resnet_four_gpu_epochs_high_acc.json" \
# 						-ep=30
						
mpiexec -n=4 python compress_model_parallel.py \
						-bs=256 \
						-im=64 \
						-sp=".summarys/resnet/cifar10_4_epochs_high_acc_freeze/" \
						-tp="resnet_four_gpu_epochs_high_acc_freeze.json" \
						-ep=30 \
						-fr=True

# mpiexec -n=3 python compress_model_parallel.py \
# 						-bs=256 \
# 						-im=64 \
# 						-sp=".summarys/resnet/cifar10_parallel_3_gpu_fixed_calls/" \
# 						-tp="resnet_three_gpu_cifar10_fixed_calls.json" \
# 						-ep=30

# mpiexec -n=2 python compress_model_parallel.py \
# 						-bs=256 \
# 						-im=64 \
# 						-sp=".summarys/resnet/cifar10_parallel_2_gpu_fixed_calls/" \
# 						-tp="resnet_two_gpu_cifar10_fixed_calls.json" \
# 						-ep=30
                        
# mpiexec -n=1 python compress_model_parallel.py \
# 						-bs=256 \
# 						-im=64 \
# 						-sp=".summarys/resnet/cifar10_parallel_1_gpu_fixed_calls/" \
# 						-tp="resnet_one_gpu_cifar10_fixed_calls.json" \
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
