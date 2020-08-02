#Data Set Params
BATCH_SIZE=32
IMAGE_SIZE=224
#TRAIN_SIZE=1034   ## orignal beans size
#TRAIN_SIZE=2152   ## scrapped dataset size
TRAIN_SIZE=3186
TEST_SIZE=128
DATA_SET="bean_scrap"
CLASSES=3

#Model Params
MODEL_PATH="../vgg19_beans.h5"
TARGET_FILE="../targets_vgg19.json"
ARCH=vgg19

#Training Params
EPOCHS=30
TEST_MULTIPLER=1
FREEZE=False

SUMMARY_PATH="/tf/notebooks/summarys/beans/vgg19/augment/"
TIMING_PATH="/tf/notebooks/timing_info/beans/vgg19/augment.json"

mpiexec -n=4 python ../compress_model_parallel.py \
						-bs=$BATCH_SIZE \
						-im=$IMAGE_SIZE \
						-ds=$DATA_SET \
						-ts=$TRAIN_SIZE \
						-vs=$TEST_SIZE \
						-ar=$ARCH \
						-ep=$EPOCHS \
						-nc=$CLASSES \
						-tf=$TARGET_FILE \
						-tp=$TIMING_PATH \
						-fr=$FREEZE \
						-mp=$MODEL_PATH 
#						-tm=$TEST_MULTIPLER \ 
#						-sp=$SUMMARY_PATH \

FREEZE=True

SUMMARY_PATH="/tf/notebooks/summarys/beans/vgg19/augment_freeze/"
TIMING_PATH="/tf/notebooks/timing_info/beans/vgg19/augment_freeze.json"

mpiexec -n=4 python ../compress_model_parallel.py \
						-bs=$BATCH_SIZE \
						-im=$IMAGE_SIZE \
						-ds=$DATA_SET \
						-ts=$TRAIN_SIZE \
						-vs=$TEST_SIZE \
						-ar=$ARCH \
						-ep=$EPOCHS \
						-nc=$CLASSES \
						-tf=$TARGET_FILE \
						-tp=$TIMING_PATH \
						-fr=$FREEZE \
						-mp=$MODEL_PATH 
#						-tm=$TEST_MULTIPLER \ 
#						-sp=$SUMMARY_PATH \

FREEZE=False
TRAIN_SIZE=1034   ## orignal beans size
DATA_SET="beans"
SUMMARY_PATH="/tf/notebooks/summarys/beans/vgg19/no_augment/"
TIMING_PATH="/tf/notebooks/timing_info/beans/vgg19/no_augment.json"

mpiexec -n=4 python ../compress_model_parallel.py \
						-bs=$BATCH_SIZE \
						-im=$IMAGE_SIZE \
						-ds=$DATA_SET \
						-ts=$TRAIN_SIZE \
						-vs=$TEST_SIZE \
						-ar=$ARCH \
						-ep=$EPOCHS \
						-nc=$CLASSES \
						-tf=$TARGET_FILE \
						-tp=$TIMING_PATH \
						-fr=$FREEZE \
						-mp=$MODEL_PATH 
#						-tm=$TEST_MULTIPLER \ 
#						-sp=$SUMMARY_PATH \
FREEZE=True
TRAIN_SIZE=1034   ## orignal beans size
DATA_SET="beans"
SUMMARY_PATH="/tf/notebooks/summarys/beans/vgg19/no_augment_freeze/"
TIMING_PATH="/tf/notebooks/timing_info/beans/vgg19/no_augment_freeze.json"

mpiexec -n=4 python ../compress_model_parallel.py \
						-bs=$BATCH_SIZE \
						-im=$IMAGE_SIZE \
						-ds=$DATA_SET \
						-ts=$TRAIN_SIZE \
						-vs=$TEST_SIZE \
						-ar=$ARCH \
						-ep=$EPOCHS \
						-nc=$CLASSES \
						-tf=$TARGET_FILE \
						-tp=$TIMING_PATH \
						-fr=$FREEZE \
						-mp=$MODEL_PATH 
#						-tm=$TEST_MULTIPLER \ 
#						-sp=$SUMMARY_PATH \


