#!/bin/bash
docker run -it --name=sigapiautocraft-$RANDOM -v $DIR:/sigapi -e DISPLAY=unix$DISPLAY sigapiautocraft:latest /sigapi/demo_venv.sh
