folder=$(pwd)
docker run -p 8888:8888  -v $folder:/tf/notebooks  -v /home/cc/my_point/:/tf/notebooks/mounting_point --gpus all -it colab bash
