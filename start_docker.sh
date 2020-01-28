folder=$(pwd)
docker run -p 8888:8888 -v $folder:/tf/notebooks --gpus all -it colab bash
