#!/bin/bash

conda create -y -n evokg python=3.7
conda activate evokg

# assumes CUDA 11.1
conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c dglteam "dgl-cuda11.1<0.7"
conda install -y pytorch-scatter -c rusty1s
conda install -y tqdm dill
conda install -y pandas