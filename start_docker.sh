folder=$(pwd)
docker run -p 8888:8888 -v $folder:/tf/notebooks -v /home/cody/tensorflow_datasets:/root/tensorflow_datasets --gpus all -it colab bash
