dataset=chengdu

# export META_PATH=./cache
export META_PATH=/home/liangxinjian/remote-disk/cache
export DATASET_PATH=../datasets/UniTE/$dataset

# python data.py --name $dataset -t trajimage-1 -g
python data.py --name $dataset -t trip -g
python data.py --name $dataset -t ksegsimidx-1000-10000 -g