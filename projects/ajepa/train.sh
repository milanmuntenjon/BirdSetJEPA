#!/usr/bin/env bash


TF_CPP_MIN_LOG_LEVEL=3 python birdset/train.py --config-path '../projects/ajepa/configs' --config-dir 'configs' "$@"