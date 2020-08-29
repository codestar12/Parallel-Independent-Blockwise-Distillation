IMAGE_SIZE=64
BATCH_SIZE=512
EPOCHS=1
SUMMARY_PATH="/tmp/"
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100.json"
ARCH="resnet"
MODEL_PATH="/tf/notebooks/base_model_cifar100_resnet34.h5"
CLASSES=100
DATA_SET="cifar100"

echo "Running ReseNet"

python compress_model_dask.py -bs=$BATCH_SIZE -tm=10 -im=$IMAGE_SIZE \
    -sp=$SUMMARY_PATH -ep=$EPOCHS -ar=$ARCH -tp=$TIMING_PATH \
    -nc=$CLASSES -ds=$DATA_SET -mp=$MODEL_PATH  

