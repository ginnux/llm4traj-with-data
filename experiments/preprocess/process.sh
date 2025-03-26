dataset=chengdu

# export META_PATH=./cache
export META_PATH=/home/liangxinjian/remote-disk/cache
export DATASET_PATH=../datasets/UniTE/$dataset

export CUDA_VISIBLE_DEVICES=1

# python data.py --name $dataset -t coccur-60
python data.py --name $dataset -t road2vec-60-128