#!/bin/bash
# get CNN models
mkdir models
cd models
wget http://www.di.ens.fr/willow/research/p-cnn/download/models.tar && tar -xvf models.tar
rm models.tar
#get ResNet PreTrained Model
wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat

# get few video examples from JHMDB
cd ..
wget http://www.di.ens.fr/willow/research/p-cnn/download/JHMDB.tar && tar -xvf JHMDB.tar
rm JHMDB.tar


