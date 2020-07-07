#!/bin/bash

#Four gpu measurement 
echo date
echo start 4 gpu measuring 
./CPU_GPU_Profiling/cpu/cpuLogToFile $1 "resnet_one_GPU" $2 &  
CPU_PID=$!
python ./CPU_GPU_Profiling/gpu/gpuProfiling.py $1 "resnet_one_GPU" $2 &
GPU_PID=$!

mpiexec -n=1 python compress_model_parallel.py \
                        -bs=256 \
                        -im=64 \
                        -sp=".summarys/resnet/cifar10_1_epochs/" \
                        -tp="resnet_one_gpu_epochs.json" \
                        -ep=16


kill -9 $CPU_PID
kill -9 $GPU_PID
echo end 4 gpu measuring 
echo date 