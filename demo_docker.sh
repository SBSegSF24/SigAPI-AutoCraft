#!/bin/bash
sudo docker build -t sigapiautocraft:latest
docker run -it --name=sigapiautocraft-$RANDOM -v /sigapi -e DISPLAY=unix$DISPLAY sigapiautocraft:latest /sigapi/demo_venv.sh
