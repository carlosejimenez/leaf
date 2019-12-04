#!/bin/sh
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate leaf
cd ./data/femnist
./preprocess.sh -s niid --sf 0.5 -k 0 -t sample --smplseed 1549786595 --spltseed 1549786796