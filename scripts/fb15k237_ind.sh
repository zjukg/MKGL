#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237_ind.yaml --version v1 > logs/fb15k237_ind_v1.log 2>&1
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237_ind.yaml --version v2 > logs/fb15k237_ind_v2.log 2>&1
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237_ind.yaml --version v3 > logs/fb15k237_ind_v3.log 2>&1
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237_ind.yaml --version v4 > logs/fb15k237_ind_v4.log 2>&1