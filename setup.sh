#!/bin/sh
conda create -n leaf python=3.6 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate leaf
conda install tensorflow=1.12.0 -y 
conda install numpy scipy Pillow matplotlib jupyter pandas nomkl -y
cd ./data/femnist
./preprocess.sh -s niid --sf 0.2 -k 0 -t sample
