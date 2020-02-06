FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install jupyterlab
RUN pip install jupyter_http_over_ws
RUN pip install tensorflow-datasets
RUN pip install matplotlib
RUN jupyter serverextension enable --py jupyter_http_over_ws
