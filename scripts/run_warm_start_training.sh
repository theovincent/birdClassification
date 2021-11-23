#!/bin/bash

train -c -m resnet -psw animals/resnet_21.pth -pd segmentation_from_network -4D -bs 32 -ne 60 -lr 0.0005 -po segmentation_from_network_start_animals

cp output/segmentation_from_network_start_animals/resnet.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_network_start_gt/
cp output/segmentation_from_network_start_animals/resnet_$(get_best_model -m resnet -po segmentation_from_network_start_animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_network_start_animals/
rm output/segmentation_from_network_start_animals/resnet*


train -c -m densenet -psw animals/densenet_19.pth -pd segmentation_from_network -4D -bs 32 -ne 60 -lr 0.0005 -po segmentation_from_network_start_animals

cp output/segmentation_from_network_start_animals/densenet.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_network_start_gt/
cp output/segmentation_from_network_start_animals/densenet_$(get_best_model -m densenet -po segmentation_from_network_start_animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_network_start_animals/
rm output/segmentation_from_network_start_animals/densenet*


train -c -m vgg -psw animals/vgg_19.pth -pd segmentation_from_network -4D -bs 32 -ne 60 -lr 0.0005 -po segmentation_from_network_start_animals

cp output/segmentation_from_network_start_animals/vgg.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_network_start_gt/
cp output/segmentation_from_network_start_animals/vgg_$(get_best_model -m vgg -po segmentation_from_network_start_animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/segmentation_from_network_start_animals/
rm output/segmentation_from_network_start_animals/vgg*
