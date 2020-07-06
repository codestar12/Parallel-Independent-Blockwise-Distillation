#!/bin/bash

# #Four gpu measurement 
# echo date
# echo start 4 gpu measuring 
# ./CPU_GPU_Profiling/cpu/cpuLogToFile $1 "vgg_four_GPU" $2 &  
# CPU_PID=$!
# python ./CPU_GPU_Profiling/gpu/gpuProfiling.py $1 "vgg_four_GPU" $2 &
# GPU_PID=$!

# echo $CPU_PID

mpiexec -n=4 python compress_model_parallel.py \
                        -bs=256 \
                        -im=32 \
                        -sp=".summarys/vgg/cifar10_4_max_acc/" \
                        -tp="vgg_four_gpu_max_acc.json" \
                        -ep=30 \
                       --arch="vgg" \
                       -mp="base_model_cifar10_32_vgg16.h5" \
                       -sd="layer_schedules/vgg16/four_gpu_bin.json" \
					   -tf="targets.json" 
					   
mpiexec -n=4 python compress_model_parallel.py \
                        -bs=256 \
                        -im=32 \
                        -sp=".summarys/vgg/cifar10_4_max_acc_freeze/" \
                        -tp="vgg_four_gpu_max_acc_freeze.json" \
                        -ep=30 \
                       --arch="vgg" \
                       -mp="base_model_cifar10_32_vgg16.h5" \
                       -sd="layer_schedules/vgg16/four_gpu_bin.json" \
					   -tf="targets.json" \
					   -fr=True

                       


# kill -9 $CPU_PID
# kill -9 $GPU_PID
# echo end 4 gpu measuring 
# echo date 

#three gpu measurement 
echo date
echo start 3 gpu measuring 
# ./CPU_GPU_Profiling/cpu/cpuLogToFile $1 "vgg_three_redo_GPU" $2 &  
# CPU_PID=$!
# python ./CPU_GPU_Profiling/gpu/gpuProfiling.py $1 "vgg_three_redo_GPU" $2 &
# GPU_PID=$!

# echo $CPU_PID

# mpiexec -n=3 python compress_model_parallel.py \
#                         -bs=256 \
#                         -im=32 \
#                         -sp=".summarys/vgg/cifar10_3_redo_epochs/" \
#                         -tp="vgg_three_gpu_epochs_redo.json" \
#                         -ep=30 \
#                        --arch="vgg" \
#                        -mp="base_model_cifar10_32_vgg16.h5" \
#                        -sd="layer_schedules/vgg16/three_gpu_bin.json" \
# 					   -tf="targets.json" 

                       


# kill -9 $CPU_PID
# kill -9 $GPU_PID
# echo end 3 gpu measuring 
# echo date 


# #three gpu measurement 
# echo date
# echo start 3 gpu measuring 
# ./CPU_GPU_Profiling/cpu/cpuLogToFile $1 "vgg_three_GPU_round_robin" $2 &  
# CPU_PID=$!
# python ./CPU_GPU_Profiling/gpu/gpuProfiling.py $1 "vgg_three_GPU_round_robin" $2 &
# GPU_PID=$!

# echo $CPU_PID

# mpiexec -n=3 python compress_model_parallel.py \
#                         -bs=256 \
#                         -im=32 \
#                         -sp=".summarys/vgg/cifar10_3_epochs_round_robin/" \
#                         -tp="vgg_three_gpu_epochs_round_robin.json" \
#                         -ep=30 \
#                        --arch="vgg" \
#                        -mp="base_model_cifar10_32_vgg16.h5" \
# 					   -tf="targets.json" 

                       


# kill -9 $CPU_PID
# kill -9 $GPU_PID
# echo end 3 gpu measuring 
# echo date 


# #two gpu measurement 
# echo date
# echo start 2 gpu measuring 
# ./CPU_GPU_Profiling/cpu/cpuLogToFile $1 "vgg_two_GPU" $2 &  
# CPU_PID=$!
# python ./CPU_GPU_Profiling/gpu/gpuProfiling.py $1 "vgg_two_GPU" $2 &
# GPU_PID=$!

# echo $CPU_PID

# mpiexec -n=2 python compress_model_parallel.py \
#                         -bs=256 \
#                         -im=32 \
#                         -sp=".summarys/vgg/cifar10_2_epochs/" \
#                         -tp="vgg_two_gpu_epochs.json" \
#                         -ep=30 \
#                        --arch="vgg" \
#                        -mp="base_model_cifar10_32_vgg16.h5" \
#                        -sd="layer_schedules/vgg16/two_gpu_bin.json" \
# 					   -tf="targets.json" 

                       


# kill -9 $CPU_PID
# kill -9 $GPU_PID
# echo end 2 gpu measuring 
# echo date 

# #two gpu measurement 
# echo date
# echo start 1 gpu measuring 
# ./CPU_GPU_Profiling/cpu/cpuLogToFile $1 "vgg_one_GPU" $2 &  
# CPU_PID=$!
# python ./CPU_GPU_Profiling/gpu/gpuProfiling.py $1 "vgg_one_GPU" $2 &
# GPU_PID=$!

# echo $CPU_PID

# mpiexec -n=1 python compress_model_parallel.py \
#                         -bs=256 \
#                         -im=32 \
#                         -sp=".summarys/vgg/cifar10_1_epochs/" \
#                         -tp="vgg_one_gpu_epochs.json" \
#                         -ep=30 \
#                        --arch="vgg" \
#                        -mp="base_model_cifar10_32_vgg16.h5" \
# 					   -tf="targets.json" 

                       


# kill -9 $CPU_PID
# kill -9 $GPU_PID
# echo end 1 gpu measuring 
# echo date 

# mpiexec -n=4 python vgg_compression_parallel_32.py \
# 						-bs=512 \
# 						-sp=".summarys/vgg/cifar10_parallel_4_aval_gpu/" \
# 						-tp="four_gpu_aval_cifar10.json" \
# 						-ep=30 \
# 						-sd="layer_schedules/vgg16/4gpu_next_aval_end.json"

# mpiexec -n=4 python vgg_compression_parallel_32.py \
# 						-bs=512 \
# 						-sp=".summarys/vgg/cifar10_parallel_4_opt_last_gpu/" \
# 						-tp="four_gpu_opt_last_cifar10.json" \
# 						-ep=30 \
# 						-sd="layer_schedules/vgg16/4gpu_opt_last.json"
