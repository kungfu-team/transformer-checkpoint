#!/bin/bash

set -e

export PYTHONPATH="$HOME/Elasticity/Repo/Megatron-LM"

python create_megatron_lm_ckpt_struct.py
