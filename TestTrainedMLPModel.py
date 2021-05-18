# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:45:46 2020

@author: flora
"""
from keras.models import load_model
import ShowFunction
import Datas
print("載入模型")
model = load_model('Mnist_mlp_model_flora.h5')
prediction = model.predict_classes(Datas.test_feature_normalize)
ShowFunction.show_image_labels_predictions(Datas.test_feature, Datas.test_label, prediction,33)
