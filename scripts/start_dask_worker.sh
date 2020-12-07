folder=$(pwd)
sudo docker run --net host -p 8787:8787  -v $(pwd):/tf/notebooks --gpus all -it colab /bin/bash -c "cd /tf/notebooks; pip install -e .; dask-cuda-worker --name worker3 tcp://10.140.81.123:8786"
