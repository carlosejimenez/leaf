#!/bin/sh
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate leaf
cd ./models/
python main.py -dataset femnist -model cnn -lr 0.06 --minibatch 0.1 --clients-per-round 3 --num-rounds 2000 --seed 42