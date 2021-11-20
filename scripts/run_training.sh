#!/bin/bash

for MODEL in resnet alexnet vgg squeezenet densenet
do
    train -c -m $MODEL -pd crop_from_network -bs 8 -ne 15 -po crop_from_network

    cp output/crop_from_network/$MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network/
    cp output/crop_from_network/$MODEL\_$(get_best_model -m $MODEL -po crop_from_network).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network/
done