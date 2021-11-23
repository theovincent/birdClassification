#!/bin/bash

for MODEL in resnet vgg densenet
do
    train -c -m $MODEL -pd animals -nc 60 -bs 32 -ne 60 -lr 0.0002 -po animals

    cp output/animals/$MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/animals/
    cp output/animals/$MODEL\_$(get_best_model -m $MODEL -po animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/animals/
    rm output/animals/$MODEL*
done