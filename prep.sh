#!/bin/bash
gpu=""
sm=""
name=`nvidia-smi --query-gpu=name --format=csv --id=0`
if [[ $(echo $name | grep 3080) ]]; then
	echo "3080 Found"
	gpu="3080"
	sm="8.6"
else
	echo "No GPU"
fi 
if [[ $gpu ]]; then
	echo "running for $gpu ..."
	pip install progress
	pip uninstall torch -y
	cd ../pytorch
	../build.sh $sm --cmake
	cd ../vision-sweep
	pkill python
	echo "running v8 heuristic mode b"
	CUDNN_V8_API_ENABLED=1 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=1 python sweep.py --output v8heurb_"$gpu"_100.csv --sku $gpu
	pkill python
	echo "running v8 bench"
	CUDNN_V8_API_ENABLED=1 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=0 python sweep.py --output v8bench_"$gpu"_100.csv --sku $gpu --bench
	pkill python
	echo "running v7 bench"
	CUDNN_V8_API_ENABLED=0 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=0 python sweep.py --output v7bench_"$gpu"_100.csv --sku $gpu --bench

else
	echo "No gpu found, not running ..."
fi
