# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:58:49 2020

@author: flora
"""
from keras.datasets import mnist
(train_feature,train_label),\
(test_feature,test_label) = mnist.load_data()
import ShowFunction
    #圖片轉一維vector(784)
train_feature_vector = train_feature.reshape(len(train_feature),784).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')
#print(train_feature_vector.shape,test_feature_vector.shape)
#print(train_feature_vector[0])
    #標準化(0~1)
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255
#print(train_feature_normalize[0])
#print(train_label[0:5])
    #LABEL轉ONE-HOT
from keras.utils import np_utils
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)
#print(train_label_onehot[0:5])
