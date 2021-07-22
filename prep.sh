#!/bin/bash
gpu=""
sm=""
name=`nvidia-smi --query-gpu=name --format=csv --id=0`
if [[ $(echo $name | grep '3080') ]]; then
	echo "3080 Found"
	gpu="3080"
	sm="8.6"
elif [[ $(echo $name | grep '3090') ]]; then
	echo "3090 Found"
	gpu="3090"
	sm="8.6"
elif [[ $(echo $name | grep 'A100') ]]; then
	echo "A100 Found"
	gpu="A100"
	sm="8.0"
elif [[ $(echo $name | grep 'V100') ]]; then
	echo "V100 Found"
	gpu="V100"
	sm="7.0"
elif [[ $(echo $name | grep 'A30') ]]; then
	echo "A30 Found"
	gpu="A30"
	sm="8.0"
elif [[ $(echo $name | grep 'A6000') ]]; then
	echo "A6000 Found"
	gpu="A6000"
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
	for zoo in [ 'timm' 'torchvision' ]
	do
		echo "running v8 heuristic mode b (" "$zoo" ")"
		CUDNN_V8_API_ENABLED=1 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=1 python sweep.py --"$zoo" --train --output train_"$zoo"_v8heurb_"$gpu"_100.csv --sku $gpu
		pkill python
		echo "running v8 bench (" "$zoo" ")"
		CUDNN_V8_API_ENABLED=1 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=0 python sweep.py --"$zoo" --train --output train_"$zoo"_v8bench_"$gpu"_100.csv --sku $gpu --bench
		pkill python
		echo "running v8 heur (" "$zoo" ")"
		CUDNN_V8_API_ENABLED=1 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=0 python sweep.py --"$zoo" --train --output train_"$zoo"_v8heur_"$gpu"_100.csv --sku $gpu
		pkill python
		echo "running v7 bench (" "$zoo" ")"
		CUDNN_V8_API_ENABLED=0 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=0 python sweep.py --"$zoo" --train --output train_"$zoo"_v7bench_"$gpu"_100.csv --sku $gpu --bench
		echo "running v7 heur (" "$zoo" ")"
		CUDNN_V8_API_ENABLED=0 CUDNN_V8_API_DEBUG=1 USE_HEURISTIC_MODE_B=0 python sweep.py --"$zoo" --train --output train_"$zoo"_v7heur_"$gpu"_100.csv --sku $gpu
	done
else
	echo "No gpu found, not running ..."
fi
