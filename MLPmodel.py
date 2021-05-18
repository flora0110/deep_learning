# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:16:14 2020

@author: flora
"""
import Datas
import ShowFunction
from keras.models import Sequential
model = Sequential()
from keras.layers import Dense
model.add(Dense(units = 256,#隱藏層神經元數目有256
                input_dim = 784, #輸入層
                kernel_initializer='normal',#使用常態分佈的亂數, 初始化weight and bias
                activation='relu' #激勵函式為relu
                #tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
                #x tensor/variable, alpha governs the slope for valus lower than the threshold
                #/relu will return max(x, 0)/max_value 可以改掉 max(x, 0)的0 /threshold the value below the threshold will be damped or set to zero
                
    ))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'
                #The elements of the output vector are in range (0, 1) and sum to 1.
                #The softmax of each vector x is computed as exp(x) / tf.reduce_sum(exp(x)).
                ))
model.compile(loss='categorical_crossentropy',#損失函數
              optimizer='adam',#最佳化方法
              metrics=['accuracy']#評估模型方式
              )
train_history = model.fit(x=Datas.train_feature_normalize,
                          y=Datas.train_label_onehot,
                          validation_split=0.2,#將訓練資料保留20%當驗證資料 剩下80%為訓練資料
                          epochs=10,batch_size=200,verbose=2)
#epochs 訓練次數 /batch_size 每次讀取多少資料 /verbose 是否顯示訓練過程 0不顯示 ,1詳細顯示, 2簡易顯示
scores=model.evaluate(Datas.test_feature_normalize,Datas.test_label_onehot)
print('\n準確率 =',scores[1])#scores[0] is loss funtion's difference
prediction=model.predict_classes(Datas.test_feature_normalize)
ShowFunction.show_image_labels_predictions(Datas.test_feature, Datas.test_label, prediction, 0)
model.save('Mnist_mlp_model_flora.h5')