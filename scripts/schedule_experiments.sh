mpiexec -n=4 python vgg_compression_parallel_32.py \
						-bs=512 \
						-sp=".summarys/vgg/cifar10_parallel_4_aval_gpu/" \
						-tp="four_gpu_aval_cifar10.json" \
						-ep=30 \
						-sd="layer_schedules/vgg16/4gpu_next_aval_end.json"

mpiexec -n=4 python vgg_compression_parallel_32.py \
						-bs=512 \
						-sp=".summarys/vgg/cifar10_parallel_4_opt_last_gpu/" \
						-tp="four_gpu_opt_last_cifar10.json" \
						-ep=30 \
						-sd="layer_schedules/vgg16/4gpu_opt_last.json"
