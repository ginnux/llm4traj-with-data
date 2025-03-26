dataset=chengdu

export META_PATH=./cache
export DATASET_PATH=../datasets/UniTE/$dataset

python main.py -c experiments/TremBR/chengdu.json --cuda 1