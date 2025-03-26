dataset=chengdu

# export META_PATH=./cache
export META_PATH=/home/liangxinjian/remote-disk/cache
export DATASET_PATH=../datasets/UniTE/$dataset

python data.py --name $dataset -t trip
python data.py --name $dataset -t quadkey-5
python data.py --name $dataset -t classbatch
python data.py --name $dataset -t timefea