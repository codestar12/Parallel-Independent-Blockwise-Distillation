#mpiexec -n=4 python vgg_compression_parallel_32.py \
#						-bs=512 \
#						-sp=".summarys/vgg/cifar10_parallel_4_gpu/" \
#						-tp="four_gpu_cifar10.txt" \
#						-ep=30
#
#mpiexec -n=3 python vgg_compression_parallel_32.py \
#						-bs=512 \
#						-sp=".summarys/vgg/cifar10_parallel_3_gpu/" \
#						-tp="three_gpu_cifar10.txt" \
#						-ep=30
#
#mpiexec -n=2 python vgg_compression_parallel_32.py \
#						-bs=512 \
#						-sp=".summarys/vgg/cifar10_parallel_2_gpu/" \
#						-tp="two_gpu_cifar10.txt" \
#						-ep=30
#
#python vgg_compression_32.py \
#			-bs=512 \
#			-sp=".summarys/vgg/cifar10/" \
#			-tp="cifar10.txt" \
#			-ep=30

mpiexec -n=4 python vgg_compression_parallel_32.py \
						-bs=512 \
						-sp=".summarys/vgg/cifar10_early_stop_parallel_4_gpu/" \
						-tp="four_gpu_cifar10_earlystop.txt" \
						-ep=30 \
						-es=True

mpiexec -n=3 python vgg_compression_parallel_32.py \
						-bs=512 \
						-sp=".summarys/vgg/cifar10_early_stop_parallel_3_gpu/" \
						-tp="three_gpu_cifar10_earlystop.txt" \
						-ep=30 \
						-es=True

mpiexec -n=2 python vgg_compression_parallel_32.py \
						-bs=512 \
						-sp=".summarys/vgg/cifar10_early_stop_parallel_2_gpu/" \
						-tp="two_gpu_cifar10_earlystop.txt" \
						-ep=30 \
						-es=True

python vgg_compression_32.py \
			-bs=512 \
			-sp=".summarys/vgg/cifar10_early_stop/" \
			-tp="cifar10_earlystop.txt" \
			-ep=30 \
			-es=True
