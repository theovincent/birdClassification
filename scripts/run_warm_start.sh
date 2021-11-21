#!/bin/bash

BEST_MODEL=

train -c -m $BEST_MODEL -psw crop_from_gt/$BEST_MODEL\_$(get_best_model -m $BEST_MODEL -po crop_from_gt).pth -pd crop_from_network -bs 32 -ne 80 -lr 0.0005 -po crop_from_network_start_gt

cp output/crop_from_network_start_gt/$BEST_MODEL.feather /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_gt/
cp output/crop_from_network_start_gt/$BEST_MODEL\_$(get_best_BEST_MODEL -m $BEST_MODEL -po crop_from_network_start_gt).pth /content/Drive/MyDrive/MVA/ObjectRecognition/birdClassification/output/crop_from_network_start_gt/
rm output/crop_from_network_start_gt/$BEST_MODEL*