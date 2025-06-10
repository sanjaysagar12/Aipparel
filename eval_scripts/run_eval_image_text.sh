#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/root/aipparel/Aipparel/
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/run.py \
 experiment.project_name=AIpparel \
 experiment.run_name=eval_image_text \
 'data_wrapper.dataset.sampling_rate=[0,0,0,1,0]' \
 pre_trained=/root/aipparel/models/aipparel_pretrained.pth \
 evaluate=True \
 --config-name aipparel --config-path ../configs