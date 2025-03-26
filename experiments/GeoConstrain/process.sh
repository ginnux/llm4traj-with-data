dataset=chengdu

# export META_PATH=./cache
export META_PATH=/home/liangxinjian/remote-disk/cache
export DATASET_PATH=../datasets/UniTE/$dataset

python data.py --name $dataset -t slice-10
python data.py --name $dataset -t transprob