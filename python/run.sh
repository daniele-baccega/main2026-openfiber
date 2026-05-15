#!/bin/bash

docker run -u $(id -u):$(id -g) -v $(pwd)/..:/app/openfiber_data --gpus all --rm openfiber:latest /usr/bin/bash -c 'export MPLCONFIGDIR=/tmp/.matplotlib && cd /app/openfiber_data/python && python3 main.py "$@"' _ "$@"