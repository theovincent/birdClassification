#!/bin/bash

for MODEL in "resnet" "alexnet" "vgg" "squeezenet" "densenet"
do
    train -m $MODEL -pd crop_from_gt -e 20 -o crop_from_gt
done