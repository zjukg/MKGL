#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/wn18rr_ind.yaml --version v1 > logs/wn8rr_ind_v1.log 2>&1
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/wn18rr_ind.yaml --version v2 > logs/wn8rr_ind_v2.log 2>&1
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/wn18rr_ind.yaml --version v3 > logs/wn8rr_ind_v3.log 2>&1
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/wn18rr_ind.yaml --version v4 > logs/wn8rr_ind_v4.log 2>&1