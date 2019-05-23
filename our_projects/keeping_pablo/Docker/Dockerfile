FROM tensorflow/tensorflow:1.13.1-py3
RUN mkdir app
RUN apt-get update 
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install pygame
RUN pip install opencv-python
WORKDIR /app
