dataset=chengdu

# export META_PATH=./cache
export META_PATH=/home/liangxinjian/remote-disk/cache
export DATASET_PATH=../datasets/UniTE/$dataset

python main.py -c experiments/GM-VSAE/chengdu.json --cuda 1