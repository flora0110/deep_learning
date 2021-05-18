# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:19:07 2020

@author: flora
"""

#print(len(train_feature),len(train_label))
#print(train_feature.shape,train_label.shape)
import matplotlib.pyplot as plt
from keras.datasets import mnist
(train_feature,train_label),\
(test_feature,test_label) = mnist.load_data()
"""def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2) #數字照片大小
    plt.imshow(image, cmap='binary') #黑白灰階顯示
    plt.show()
"""
#show_image(train_feature[0])
#print(train_label[0])
def show_image_labels_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12,14)
    if num>25:num=25
    for i in range(num):
        ax=plt.subplot(5,5,i+1)
        #顯示黑白照片
        ax.imshow(images[start_id],cmap='binary')
        #有ai預測結果資料, 才在標題顯示預測結果
        if(len(predictions)>0):
            title = 'ai =' +str(predictions[i])
            #預測正確顯使(0), 錯誤顯示(x)
            title += ('(0)' if predictions[i]==labels[i] else'(x)')
            title +='\nlabel ='+str(labels[i])
        else:
            title = 'label =' + str(labels[i])
        #x,y 軸不顯示刻度
        ax.set_title(title,fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
        start_id+=1
    plt.show()
#show_image_labels_predictions(train_feature, train_label, [], 0,10)