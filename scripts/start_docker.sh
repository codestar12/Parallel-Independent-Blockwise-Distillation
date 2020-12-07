folder=$(pwd)
docker run --net host -p 8787:8787  -v $folder:/tf/notebooks --gpus all -it colab bash
