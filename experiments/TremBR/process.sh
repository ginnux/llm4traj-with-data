dataset=chengdu

export META_PATH=./cache
export DATASET_PATH=../datasets/UniTE/$dataset

python data.py --name $dataset -t trip
python data.py --name $dataset -t trip-traj2vectime
python data.py --name $dataset -t tte