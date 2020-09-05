IMAGE_SIZE=64
BATCH_SIZE=256
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

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR

#EPOCHS=60
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep60_t30_lr02_flr006.json"
#TUNE_EPOCHS=30
#LR=.02
#FLR=.006

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR

#EPOCHS=60
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep60_t30_lr002_flr006.json"
#TUNE_EPOCHS=30
#LR=.002
#FLR=.006

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR

#EPOCHS=90
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep90_t30_lr002_flr006.json"
#TUNE_EPOCHS=30
#LR=.002
#FLR=.006

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR


#EPOCHS=30
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t30_lr02_flr06.json"
#TUNE_EPOCHS=30
#LR=.02
#FLR=.06

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR


#EPOCHS=30
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t60_lr02_flr06.json"
#TUNE_EPOCHS=60
#LR=.02
#FLR=.06

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR

#EPOCHS=30
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep30_t90_lr02_flr06.json"
#TUNE_EPOCHS=90
#LR=.02
#FLR=.06

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR

# EPOCHS=89
# TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep90_t90_lr02_flr06.json"
# TUNE_EPOCHS=90
# LR=.02
# FLR=.006

# echo "Running ResNet"

# python compress_model_dask.py \
#      -flr $FLR \
#      -bs $BATCH_SIZE \
#      -tm 1 \
#      -im $IMAGE_SIZE \
#      -sp $SUMMARY_PATH \
#      -ep $EPOCHS \
#      -ar $ARCH \
#      -tp $TIMING_PATH \
#      -nc $CLASSES \
#      -ds $DATA_SET \
#      -mp $MODEL_PATH \
#      -fe $TUNE_EPOCHS \
#      -lr $LR

# EPOCHS=61
# TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep60_t90_lr002_flr006.json"
# TUNE_EPOCHS=90
# LR=.002
# FLR=.006

# echo "Running ResNet"

# python compress_model_dask.py \
#      -flr $FLR \
#      -bs $BATCH_SIZE \
#      -tm 1 \
#      -im $IMAGE_SIZE \
#      -sp $SUMMARY_PATH \
#      -ep $EPOCHS \
#      -ar $ARCH \
#      -tp $TIMING_PATH \
#      -nc $CLASSES \
#      -ds $DATA_SET \
#      -mp $MODEL_PATH \
#      -fe $TUNE_EPOCHS \
#      -lr $LR

# EPOCHS=61
# TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep60_t90_lr002_flr006_threshold_5_ES_True.json"
# TUNE_EPOCHS=90
# LR=.002
# FLR=.006

# echo "Running ResNet"

# python compress_model_dask.py \
#      -flr $FLR \
#      -es True \
#      -bs $BATCH_SIZE \
#      -tm 1 \
#      -im $IMAGE_SIZE \
#      -sp $SUMMARY_PATH \
#      -ep $EPOCHS \
#      -ar $ARCH \
#      -tp $TIMING_PATH \
#      -nc $CLASSES \
#      -ds $DATA_SET \
#      -mp $MODEL_PATH \
#      -fe $TUNE_EPOCHS \
#      -lr $LR

EPOCHS=180
TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep180_t180_lr01_flr0001_threshold_4_ES_False.json"
TUNE_EPOCHS=180
LR=.01
FLR=.0001

echo "Running ResNet"

python compress_model_dask.py \
     -flr $FLR \
     -es False \
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

#EPOCHS=120
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep120_t120_lr001_flr001_threshold_15_ES_True_freeze.json"
#TUNE_EPOCHS=120
#LR=.001
#FLR=.0001

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -es True \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR \
#     -fr True

#EPOCHS=120
#TIMING_PATH="/tf/notebooks/timing_info/dask/resnet34_cifar100_ep120_t120_lr0001_flr001_threshold_15_ES_True_freeze.json"
#TUNE_EPOCHS=120
#LR=.001
#FLR=.0001

#echo "Running ResNet"

#python compress_model_dask.py \
#     -flr $FLR \
#     -es True \
#     -bs $BATCH_SIZE \
#     -tm 1 \
#     -im $IMAGE_SIZE \
#     -sp $SUMMARY_PATH \
#     -ep $EPOCHS \
#     -ar $ARCH \
#     -tp $TIMING_PATH \
#     -nc $CLASSES \
#     -ds $DATA_SET \
#     -mp $MODEL_PATH \
#     -fe $TUNE_EPOCHS \
#     -lr $LR \
#     -fr True
