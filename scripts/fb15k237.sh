#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237.yaml > logs/fb15k237.log 2>&1