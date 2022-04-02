# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:37:29 2022

@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

Xrange,Yrange=128,128
X, Y = np.linspace(0, Xrange-1,Xrange,dtype=int), np.linspace(0, Yrange-1,Yrange,dtype=int)
X2, Y2 = np.linspace(0, 2*Xrange-1,2*Xrange,dtype=int), np.linspace(0, 2*Yrange-1,2*Yrange,dtype=int)

#------------------------------------------------------

import numpy.fft as nf
import tensorflow as tf

# 高斯方法生成
num=1
N=np.arange(0, num) #高斯数量
def Gaussian(x,y,x0,y0,rx,ry,amp,pha):
    return amp*np.exp(-(x-x0)*(x-x0)/rx/rx-(y-y0)*(y-y0)/ry/ry+1j*pha)

x0=np.random.randint(40,Xrange-40,size=(num)) #位置x0
y0=np.random.randint(40,Yrange-40,size=(num)) #位置y0
rx=np.random.normal(loc=20.0, scale=10.0, size=(num)) #束腰rx 大小范围待定
ry=np.random.normal(loc=25.0, scale=10.0, size=(num)) #束腰ry
Amp=np.abs(np.random.normal(loc=1.0, scale=0.5, size=(num))) #高斯幅度
PhaR=np.random.rand(num)*2*np.pi # 随机相位 0，2pi

SurfR=np.zeros((Xrange,Yrange, num),dtype=complex)
Surf=np.zeros((Xrange,Yrange),dtype=complex)
for n in N:
    for x,y in product (X, Y):
        SurfR[x,y,n]=Gaussian(x,y,x0[n],y0[n],rx[n],rx[n],Amp[n],PhaR[n])
    print (n)
    
Surf=np.sum(SurfR,axis=2)
# Ampl=np.abs(Surf)
# Phas=np.angle(Surf)
# # plt.contourf(X,Y,Ampl, 50, cmap='rainbow')
# # plt.show()

#Ex=np.zeros((2*Xrange,2*Yrange))
Ex=np.pad(Surf,((int(Xrange/2),int(Xrange/2)),(int(Yrange/2),int(Yrange/2))),"constant",constant_values=(0, 0)) #补0
#XXY=np.broadcast_to(X2,(Xrange*2,Yrange*2))
#XYY=np.broadcast_to(Y2.T,(Xrange*2,Yrange*2))
#PShiftX=np.exp(1j*2*np.pi*Xrange/(2*Xrange+1)*XXY)
#PShiftY=np.exp(1j*2*np.pi*Yrange/(2*Yrange+1)*XYY)
#Ex=Ex*PShiftX*PShiftY
AEx=np.fft.fft2(Ex)
AEx=np.fft.fftshift(AEx)  # 谱移到中心位置
#PAEx=np.angle(AEx)

plt.contourf(X2,Y2,np.angle(AEx), 50, cmap='rainbow')
# plt.show()
# ax = plt.axes(projection='3d')
# plt.contourf(X2,Y2,PAEx, 50, cmap='rainbow')
# plt.show()






# NumSamples=1000
# tfrecord_file = '.\\PhaseImages\\PhaseTrainData.tfrecords'
# with tf.io.TFRecordWriter(tfrecord_file) as writer:
#      for i in range(NumSamples):
         
         
         
#          UnWrapPhase=Surf*np.pi  #生成相位 
#          WrapPhase=np.arctan2(np.sin(UnWrapPhase),np.cos(UnWrapPhase)) #折叠后相位
#          noise=np.random.normal(size=(Xrange,Yrange))*0.2 #噪声强度
#          WrapPhaseNoise=WrapPhase+noise #折叠后相位加随机噪声
         
#          Mask = np.round((UnWrapPhase)/(2*np.pi)) #相位阶数
         
#          print (i)
#          # plt.contourf(X,Y,Mask[:,:,0], 50, cmap='rainbow')
#          # plt.show() 
#          # plt.contourf(X,Y,MaskCode[:,:,0], 50, cmap='rainbow')  
#          # plt.show() 
#          # plt.contourf(X,Y,MaskCode[:,:,1], 50, cmap='rainbow')
#          # plt.show() 

#          #创建3d绘图区域
#          # ax = plt.axes(projection='3d')
#          # plt.contourf(X,Y,Surf, 50, cmap='rainbow')
#          # plt.show()
#          #ax = plt.axes(projection='3d')
#          # plt.contourf(X,Y,StepMask, 50, cmap='rainbow')
#          # plt.show()
#          # plt.contourf(X,Y,WrapPhase, 50, cmap='rainbow')
#          # plt.show()
#          #ax = plt.axes(projection='3d')

#          np.save(".\\PhaseImages\\TestData\\UnWrapPhase"+str(0)+'.npy',UnWrapPhase)
#          np.save(".\\PhaseImages\\TestData\\WrapPhase"+str(0)+'.npy',WrapPhase)
#          np.save(".\\PhaseImages\\TestData\\Mask"+str(0)+'.npy',Mask)
#          np.save(".\\PhaseImages\\TestData\\WrapPhaseNoise"+str(0)+'.npy',WrapPhaseNoise)
       
#          # train_datafiles.append(WrapPhasefile)
#          # train_Truthfiles.append(UnWrapPhasefile)
         
#          WrapPhase=WrapPhase.flatten().tolist()
#          Mask=Mask.flatten().tolist()  
         
# #         UnWrapPhase=UnWrapPhase.flatten().tolist()  
         
#          featureDic = {   # 建立 tf.train.Feature 字典
#             'image': tf.train.Feature(float_list=tf.train.FloatList(value=WrapPhase)),  
#             'Truth': tf.train.Feature(float_list=tf.train.FloatList(value=Mask))   
#          }
#          # 通过字典建立 Example, 序列化并写入 TFRecord 文件，登记为string数据
#          example = tf.train.Example(features=tf.train.Features(feature=featureDic)) 
#          writer.write(example.SerializeToString())   
# print("All Files are generated.")



# # 读取 TFRecord 文件
# dataset = tf.data.TFRecordDataset(tfrecord_file)    
# feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
#     'image': tf.io.FixedLenFeature([Xrange,Yrange], tf.float32),
#     'Truth': tf.io.FixedLenFeature([Xrange,Yrange], tf.float32)
# }

# def read_example(example_string): #    从TFrecord格式文件中读取数据
#     feature_dict = tf.io.parse_single_example(example_string, feature_description)
#     return feature_dict['image'], feature_dict['Truth']

# dataset = dataset.map(read_example) # 解析数据

# for index, one_element in enumerate(dataset): #检查数据
#     print(index,one_element[0].shape,one_element[1].shape)
#     plt.contourf(X,Y,one_element[0], 50, cmap='rainbow')
#     plt.show() 
#     plt.contourf(X,Y,one_element[1], 50, cmap='rainbow')
#     plt.show() 

