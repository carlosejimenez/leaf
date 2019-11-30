pip install tensorflow==1.12.0
pip install numpy scipy Pillow matplotlib jupyter pandas nomkl
cd ./data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
