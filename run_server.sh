#!/usr/bin/env bash
docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/saved_model_basic:/models/saved_model_basic" \
    -e MODEL_NAME=saved_model_basic \
    tensorflow/serving 
