#!/bin/bash

for MODEL in resnet alexnet vgg squeezenet densenet
do
    train -c -m $MODEL -pd segmentation_from_gt -4D -bs 32 -ne 80 -lr 0.0005 -po segmentation_from_gt

    cp output/segmentation_from_gt/$MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_gt/
    cp output/segmentation_from_gt/$MODEL\_$(get_best_model -m $MODEL -po segmentation_from_gt).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_gt/
    rm output/segmentation_from_gt/$MODEL*
done

MODEL=efficientnet
train -c -m $MODEL -pd segmentation_from_gt -4D -bs 8 -ne 80 -lr 0.00005 -po segmentation_from_gt

cp output/segmentation_from_gt/$MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_gt/
cp output/segmentation_from_gt/$MODEL\_$(get_best_model -m $MODEL -po segmentation_from_gt).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_gt/
rm output/segmentation_from_gt/$MODEL*