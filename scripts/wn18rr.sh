#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/wn18rr.yaml > logs/wn18rr.log 2>&1