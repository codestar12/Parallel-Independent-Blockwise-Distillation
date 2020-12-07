IMAGE_SIZE=64
BATCH_SIZE=512
EPOCHS=30
SUMMARY_PATH="/tmp/"
TIMING_PATH="/tf/notebooks/timing_info/dask/two_gpu/resnet.json"
ARCH="resnet"
MODEL_PATH="/tf/notebooks/cifar10.h5"


echo "Running ReseNet"

python compress_model_dask.py \
    -bs=$BATCH_SIZE \
    -im=$IMAGE_SIZE \
    -sp=$SUMMARY_PATH \
    -tp=$TIMING_PATH \
    --arch=$ARCH \
    -ep=$EPOCHS \
    -mp=$MODEL_PATH 

IMAGE_SIZE=32
TIMING_PATH="/tf/notebooks/timing_info/dask/two_gpu/vgg.json"
ARCH="vgg"
MODEL_PATH="/tf/notebooks/base_model_cifar10_32_vgg16.h5"

python compress_model_dask.py \
    -bs=$BATCH_SIZE \
    -im=$IMAGE_SIZE \
    -sp=$SUMMARY_PATH \
    -tp=$TIMING_PATH \
    --arch=$ARCH \
    -ep=$EPOCHS \
    -mp=$MODEL_PATH 
