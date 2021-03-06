#!/bin/bash

for MODEL in resnet alexnet vgg squeezenet densenet
do
    train -c -m $MODEL -pd crop_from_gt -bs 32 -ne 80 -lr 0.0005 -po crop_from_gt

    cp output/crop_from_gt/$MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_gt/
    cp output/crop_from_gt/$MODEL\_$(get_best_model -m $MODEL -po crop_from_gt).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_gt/
    rm output/crop_from_gt/$MODEL*
done

MODEL=efficientnet
train -c -m $MODEL -pd crop_from_gt -bs 8 -ne 80 -lr 0.00005 -po crop_from_gt

cp output/crop_from_gt/$MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_gt/
cp output/crop_from_gt/$MODEL\_$(get_best_model -m $MODEL -po crop_from_gt).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_gt/
rm output/crop_from_gt/$MODEL*