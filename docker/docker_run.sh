#!/bin/bash

#xhost +
xhost +si:localuser:$USER

docker run --gpus '"device=1","capabilities=compute,graphics,display,utility"' -it --rm \
  -p 6006:6006 \
  -p 8888:8888 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/martin/devel/github/martasbee/udacity_AIPND_project:/workspace/AIPND/repo \
  -v /home/martin/devel/data:/workspace/AIPND/data \
  nvcr.io/nvidia/pytorch:21.05-py3 bash
