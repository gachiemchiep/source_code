bash tools/dist_train.sh ../scripts/configs/efficientnet-b0.py 4

python tools/train.py ../scripts/configs/efficientnet-b0.py --gpu-id 3
python tools/train.py ../scripts/configs/resnet18.py --gpu-id 2


# current
fold 0 : efficientnet-b0   resnet18     