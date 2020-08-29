IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=21
SUMMARY_PATH="/tmp/"
TIMING_PATH="/tf/notebooks/timing_info/dask/two_gpu/resnet_imagenet.json"
ARCH="resnet"
#MODEL_PATH="/tf/notebooks/cifar10.h5"
CLASSES=1000
DATA_SET="imagenet"


echo "Running ResNet"

python compress_model_dask.py \
    -bs=$BATCH_SIZE \
    -tm=1 \
    -im=$IMAGE_SIZE \
    -sp=$SUMMARY_PATH \
    -tp=$TIMING_PATH \
    --arch=$ARCH \
    -ep=$EPOCHS \
    -nc=$CLASSES \
    -ds=$DATA_SET  

#IMAGE_SIZE=32
TIMING_PATH="/tf/notebooks/timing_info/dask/two_gpu/vgg_imagenet.json"
ARCH="vgg"
#MODEL_PATH="/tf/notebooks/base_model_cifar10_32_vgg16.h5"

python compress_model_dask.py \
    -bs=$BATCH_SIZE \
    -tm=1 \
    -im=$IMAGE_SIZE \
    -sp=$SUMMARY_PATH \
    -tp=$TIMING_PATH \
    --arch=$ARCH \
    -ep=$EPOCHS \
    -nc=$CLASSES \
    -ds=$DATA_SET  
