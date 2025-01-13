@echo off
setlocal

set PYTHONPATH=.
set TENSORBOARD_PORT=6007

echo Starting training with following parameters:
echo Batch Size: %1
echo Epochs: %2
echo Learning Rate: %3

REM Start monitoring services
start /B tensorboard --logdir=logs/tensorboard --port=%TENSORBOARD_PORT%

REM Run training
python -m core.training.train ^
    --batch-size %1 ^
    --epochs %2 ^
    --learning-rate %3

endlocal