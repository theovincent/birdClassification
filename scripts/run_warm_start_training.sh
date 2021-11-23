#!/bin/bash

train -c -m resnet -psw animals/resnet_21.pth -pd crop_from_network -bs 32 -ne 60 -lr 0.0005 -po crop_from_network_start_animals

cp output/crop_from_network_start_animals/resnet.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_animals/
cp output/crop_from_network_start_animals/resnet_$(get_best_model -m resnet -po crop_from_network_start_animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_animals/
rm output/crop_from_network_start_animals/resnet*


train -c -m densenet -psw animals/densenet_19.pth -pd crop_from_network -bs 32 -ne 60 -lr 0.0005 -po crop_from_network_start_animals

cp output/crop_from_network_start_animals/densenet.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_animals/
cp output/crop_from_network_start_animals/densenet_$(get_best_model -m densenet -po crop_from_network_start_animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_animals/
rm output/crop_from_network_start_animals/densenet*


train -c -m vgg -psw animals/vgg_19.pth -pd crop_from_network -bs 32 -ne 60 -lr 0.0005 -po crop_from_network_start_animals

cp output/crop_from_network_start_animals/vgg.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_animals/
cp output/crop_from_network_start_animals/vgg_$(get_best_model -m vgg -po crop_from_network_start_animals).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_animals/
rm output/crop_from_network_start_animals/vgg*
