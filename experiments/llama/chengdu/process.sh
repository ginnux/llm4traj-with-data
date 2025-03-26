dataset=chengdu

# export META_PATH=./cache
export META_PATH=/mnt/extent/home/liangxinjian/projects/cache
export DATASET_PATH=/mnt/extent/home/liangxinjian/projects/datasets/UniTE/$dataset

# python data.py --name $dataset -t trajimage-1 -g
python data.py --name $dataset -t trip
