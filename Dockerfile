FROM tensorflow/tensorflow:2.3.0-gpu
RUN apt-get update
RUN apt-get install -y mpich
RUN pip install mpi4py
RUN pip install ptvsd==4.3.2
RUN pip install py3nvml
RUN pip install jupyterlab
RUN pip install jupyter_http_over_ws
RUN pip install tensorflow-datasets
RUN pip install matplotlib
RUN pip install dask
RUN pip install dask-cuda
RUN pip install bokeh
RUN jupyter serverextension enable --py jupyter_http_over_ws
RUN python -m pip install jupyter-server-proxy
COPY ./CPU_GPU_Profiling/ /CPU_GPU_Profiling/ 
RUN make ./CPU_GPU_Profiling/
