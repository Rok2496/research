#!/bin/bash

# Environment setup
python -m venv venv
source venv/bin/activate

# Install dependencies based on environment
if [ "$ENVIRONMENT" = "development" ]; then
    pip install -r requirements/dev.txt
else
    pip install -r requirements/prod.txt
fi

# Create required directories
mkdir -p logs/{tensorboard,deployment}
mkdir -p models/{saved,checkpoints}
mkdir -p data/{raw,processed}

# Download pretrained weights
python -c "
import torch
import timm
model = timm.create_model('efficientnet_v2_l', pretrained=True)
torch.save(model.state_dict(), 'models/pretrained/efficientnet_v2.pth')
"

# Setup CUDA if available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "CUDA available, setting up TensorRT..."
    python -m pip install tensorrt
fi

echo "Setup completed successfully!"
