# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:51:22 2019

@author: sbtithzy
"""
import pandas as pd
import numpy as np
##导入数据 
data = pd.read_csv('1.csv')
###
X = data.iloc[:,3:13].values
y = data.iloc[:,-1].values
######第一步#####数据预处理 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencode_1 = LabelEncoder()###定义一个LabelEncodede类 
X[:,1] = labelencode_1.fit_transform(X[:,1])######将X的第二列转变为哑变量 
labelencode_2 = LabelEncoder()###定义一个LabelEncodede类 
X[:,2] = labelencode_2.fit_transform(X[:,2])######将X的第三列转变为哑变量 
####OneHotEncode
onehotencode = OneHotEncoder(categorical_features=[1])
X = onehotencode.fit_transform(X).toarray()
X = X[:,1:]
#######划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
#####特征缩放
from  sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test  = standardscaler.fit_transform(X_test)
######第二步#####构造ANN 

import keras
from keras.layers import Dense
from keras.models import Sequential
classifier = Sequential()
###添加输入层和隐藏层
###隐藏层,初始化权值为随机 ,激活函数使用线性整流函数 ,输入层的个数为11
classifier.add(Dense(units = 8,kernel_initializer='uniform',activation='relu',input_dim = 11))
###再添加一个隐藏
classifier.add(Dense(units = 8,kernel_initializer='uniform',activation='relu'))
###添加一个输出层
classifier.add(Dense(units = 1,kernel_initializer='uniform',activation='sigmoid'))
# =============================================================================
# 编译ANN
# 编译ANN的时候重要参数,optimizer,loss,这里选择随机梯度下降,损失函数选择 binary_crossentropy,
# 主要考虑输出是二分类的数据， metrics这里关注准确率 
# =============================================================================
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
# =============================================================================
# 训练数据
# batch_size=10表示10个训练数据一组一个损失和改变一次W，epochs=50,训练50次
# =============================================================================
classifier.fit(X_train,y_train,batch_size = 5,epochs = 100)

######第三步#####预测和模型评估 
# 预测
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.45)
# 评估模型 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
