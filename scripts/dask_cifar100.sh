IMAGE_SIZE=64
BATCH_SIZE=512
EPOCHS=30
SUMMARY_PATH="/tmp/"
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t30_lr02_flr006.json"
ARCH="resnet"
MODEL_PATH="/tf/notebooks/base_model_cifar100_resnet34.h5"
CLASSES=100
DATA_SET="cifar100"
TUNE_EPOCHS=30
LR=0.02
FLR=0.006

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR

EPOCHS=60
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep60_t30_lr02_flr006.json"
TUNE_EPOCHS=30
LR=.02
FLR=.006

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR

EPOCHS=60
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep60_t30_lr002_flr006.json"
TUNE_EPOCHS=30
LR=.002
FLR=.006

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR

EPOCHS=90
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep90_t30_lr002_flr006.json"
TUNE_EPOCHS=30
LR=.002
FLR=.006

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR


EPOCHS=30
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t30_lr02_flr06.json"
TUNE_EPOCHS=30
LR=.02
FLR=.06

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR


EPOCHS=30
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t60_lr02_flr06.json"
TUNE_EPOCHS=60
LR=.02
FLR=.06

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR

EPOCHS=30
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t90_lr02_flr06.json"
TUNE_EPOCHS=90
LR=.02
FLR=.06

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR

EPOCHS=90
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep90_t90_lr02_flr06.json"
TUNE_EPOCHS=90
LR=.02
FLR=.06

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -bs $BATCH_SIZE \
     -tm 1 \
     -im $IMAGE_SIZE \
     -sp $SUMMARY_PATH \
     -ep $EPOCHS \
     -ar $ARCH \
     -tp $TIMING_PATH \
     -nc $CLASSES \
     -ds $DATA_SET \
     -mp $MODEL_PATH \
     -fe $TUNE_EPOCHS \
     -lr $LR