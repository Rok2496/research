#!/bin/bash
python -m core.training.train \
    --config configs/training_config.yaml \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001