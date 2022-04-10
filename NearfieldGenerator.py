# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:37:29 2022

@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.interpolate import griddata
import numpy.fft as nf
import tensorflow as tf
'''
基本输入
'''
c = 2.998*1E8#光速
f = 1E10#频率
waveLambda = c/f#波长
k = 2*np.pi/waveLambda#k矢波数
ds = waveLambda/3#离散精度

Xrange,Yrange=128,128#范围点数
X, Y = np.linspace(0, Xrange-1,Xrange,dtype=int), np.linspace(0, Yrange-1,Yrange,dtype=int)
Xrangeext,Yrangeext = 2*Xrange,2*Yrange#扩零后的点数
Xext, Yext = np.linspace(0, Xrangeext-1,Xrangeext,dtype=int), np.linspace(0, Yrangeext-1,Yrangeext,dtype=int)
Xext2 = np.transpose(np.broadcast_to(Xext,(Xrangeext,Yrangeext)))#扩0后逐点的X坐标
Yext2 = np.broadcast_to(Xext,(Xrangeext,Yrangeext))

theta =np.pi/3
phi = np.pi/3

NumSamples = 1
tfrecord_file = '.\\Images_z_theta_phi\\ValidationData.tfrecords'
with tf.io.TFRecordWriter(tfrecord_file) as writer:
     for i in range(NumSamples):
        # 高斯方法生成
        num=5
        N=np.arange(0, num) #高斯数量
        def Gaussian(x,y,x0,y0,rx,ry,amp,pha):
            return amp*np.exp(-(x-x0)*(x-x0)/rx/rx-(y-y0)*(y-y0)/ry/ry+1j*pha)

        x0=np.random.randint(50,Xrange-48,size=(num)) #位置x0
        y0=np.random.randint(50,Yrange-48,size=(num)) #位置y0
        rx=np.random.normal(loc=20.0, scale=10.0, size=(num)) #束腰rx 大小范围待定
        ry=np.random.normal(loc=25.0, scale=10.0, size=(num)) #束腰ry
        Amp=np.abs(np.random.normal(loc=1.0, scale=0.5, size=(num))) #高斯幅度
        PhaR=np.random.rand(num)*2*np.pi # 随机相位 0，2pi
        SurfR=np.zeros((Xrange,Yrange, num),dtype=complex)
        Surf=np.zeros((Xrange,Yrange),dtype=complex)
        for n in N:
            for x,y in product (X, Y):
                SurfR[x,y,n]=Gaussian(x,y,x0[n],y0[n],rx[n],rx[n],Amp[n],PhaR[n])
        Surf=np.sum(SurfR,axis=2)    
         
        """
        坐标旋转，theta为俯仰角，phi为方位角
        """
        z0 = (np.random.rand()+0.001)*200*waveLambda#传播距离

        #旋转矩阵，Ma为从初始平面变换至旋转平面；MaI相反
        MaI = np.array([[np.cos(phi)*np.cos(theta), -np.sin(phi), np.cos(phi)*np.sin(theta)],[np.sin(phi)*np.cos(theta), np.cos(phi), np.sin(phi)*np.sin(theta)],[-np.sin(theta), 0, np.cos(theta)]])
        Ma = np.linalg.inv(MaI)
        #此处可验证旋转矩阵的正确性
        # #初始坐标系下基矢量
        # U0 = np.array([1,0,0])
        # V0 = np.array([0,1,0])
        # N0 = np.array([0,0,1])
        # #des坐标系下的变换向量
        # Ut = MaI.dot(U0)
        # Vt = MaI.dot(V0)
        # Nt = MaI.dot(N0)
        """
        补充添0
        """
        Eext = np.pad(Surf, (int(Xrange/2),int(Yrange/2)), 'constant')

        """
        傅里叶变换及频谱坐标移动到中心
        """
        AEext = np.fft.fft2(Eext)
        AEext = np.fft.fftshift(AEext)
        # #此步为将频域进行相位搬移，原因在于积分区间有限，傅里叶变换需要相位补偿
        AEext = AEext*np.exp(-1j*2*np.pi*0.5*Xext2)*np.exp(-1j*2*np.pi*0.5*Yext2)

        """
        角谱坐标系变换
        """
        #V*t为旋转平面的角谱坐标，V*为初始平面的角谱坐标
        alpha = waveLambda * ((Xext2-0.5*Xrangeext))/ds/Xrangeext
        beta = waveLambda * (Yext2 - 0.5*Yrangeext)/ds/Yrangeext
        gamma = np.sqrt(1-alpha**2-beta**2)
        np.nan_to_num(gamma,copy=False,nan = -2)

        Vxt = alpha/waveLambda
        Vyt = beta/waveLambda
        Vzt = gamma/waveLambda
        Vt = np.array([Vxt,Vyt,Vzt])
        V= np.dot(MaI,Vt.reshape(3,Xrangeext*Yrangeext)).reshape(3,Xrangeext,Yrangeext)
        Vx = V[0,:,:]
        Vy = V[1,:,:]
        Vz = V[2,:,:]
        Vxt = Vxt[:,0]
        Vyt = Vyt[0,:]
        #griddata函数参数points为[n,2]形式，第一列为x坐标值，第二列为y坐标值；AEext需要与其匹配
        points = np.array(np.meshgrid(Vxt, Vyt)).T.reshape(-1,2)
        AEext = AEext.reshape(Xrangeext*Yrangeext,1)
        AEextG = griddata(points, AEext, (Vx,Vy),method='linear').reshape(Xrangeext,Yrangeext)
        np.nan_to_num(AEextG,copy=False,nan = -2)
        AEextG = AEextG * (np.abs(Vz)/np.abs(Vzt))
        AEextG[Vz<=0.01] = 1E-19

        '''
        定义系统传播函数
        '''
        #计算旋转后原点的位置T(xt,yt,zt)
        T = Ma.dot(np.array([0,0,z0]))
        #扩0后初始坐标系的实际坐标位置
        XPos = (Xext2 - Xrangeext/2)*ds
        YPos = (Yext2 - Yrangeext/2)*ds
        #各个点的空间距离
        R = np.sqrt((XPos+T[0])**2+(YPos+T[1])**2+T[2]**2)
        #空域系统函数的空间分布及其傅里叶变换
        h = np.exp(1j*k*R)*T[2]*(1+1j*k*R)/(2*np.pi*R**3) 
        H = np.fft.fft2(h)

        '''
        谱域相乘，能量补偿及计算空域场分布
        '''
        #将谱域再分配至四角可与传播函数直接相乘
        AEextG = np.fft.fftshift(AEextG)
        #IFFT变回空域场分布
        Eresult = np.fft.ifft2(AEextG*H)
        #怕斯维尔定理能量补偿
        Powerh = ds**2*np.sum(np.abs(h)**2)
        PowerH = (1/Xrangeext/ds)*(1/Yrangeext/ds)*np.sum(np.abs(H)**2)
        Eresult = Eresult*np.sqrt(Powerh/PowerH)
        #截取初始分布的面积
        Eresult = Eresult[int(Xrange/2):int(Xrange/2)+Xrange,int(Yrange/2):int(Yrange/2)+Yrange]
        
        
        print (i)


        Surf2Write = np.zeros([Xrange,Yrange,5],dtype = np.float32)
        Surf2Write[:,:,0] = np.abs(Surf)
        Surf2Write[:,:,1] = np.angle(Surf)
        Surf2Write[:,:,2] = z0
        Surf2Write[:,:,3] = theta
        Surf2Write[:,:,4] = phi
        np.save(".\\Images_z_theta_phi\\validation_data\\Source\\source"+str(i)+'.npy',Surf2Write)
        Surf2Write = Surf2Write.flatten().tolist()
        Eresult2Write = np.zeros([Xrange,Yrange,2],dtype = np.float32)
        Eresult2Write[:,:,0] = np.abs(Eresult)
        Eresult2Write[:,:,1] = np.angle(Eresult)
        np.save(".\\Images_z_theta_phi\\validation_data\\Target\\target"+str(i)+'.npy',Eresult2Write) 
        Eresult2Write =Eresult2Write.flatten().tolist()     
        featureDic = {   # 建立 tf.train.Feature 字典
           'Source': tf.train.Feature(float_list=tf.train.FloatList(value=Surf2Write)), 
           'Target': tf.train.Feature(float_list=tf.train.FloatList(value=Eresult2Write))
        }
        # 通过字典建立 Example, 序列化并写入 TFRecord 文件，登记为string数据
        example = tf.train.Example(features=tf.train.Features(feature=featureDic)) 
        writer.write(example.SerializeToString())   
print("All Files are generated.")

# dataset = tf.data.TFRecordDataset(tfrecord_file)    
# feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
#     'Source': tf.io.FixedLenFeature([Xrange,Yrange,5], tf.float32),
#     'Target': tf.io.FixedLenFeature([Xrange,Yrange,2], tf.float32)
# }

# def read_example(example_string): #    从TFrecord格式文件中读取数据
#     feature_dict = tf.io.parse_single_example(example_string, feature_description)
#     return feature_dict['Source'], feature_dict['Target']

# dataset = dataset.map(read_example) # 解析数据

# for index, one_element in enumerate(dataset): #检查数据
#     print(index,one_element[0].shape,one_element[1].shape)
#     plt.contourf(X,Y,np.abs(Surf), 50, cmap='rainbow')
#     plt.show() 
#     plt.contourf(X,Y,np.angle(Surf), 50, cmap='rainbow')
#     plt.show()  
#     plt.contourf(X,Y,np.abs(Eresult), 50, cmap='rainbow')
#     plt.show() 
#     plt.contourf(X,Y,np.angle(Eresult), 50, cmap='rainbow')
#     plt.show()  
#     plt.contourf(X,Y,one_element[0][:,:,0], 50, cmap='rainbow')
#     plt.show() 
#     plt.contourf(X,Y,one_element[0][:,:,1], 50, cmap='rainbow')
#     plt.show() 
#     plt.contourf(X,Y,one_element[1][:,:,0], 50, cmap='rainbow')
#     plt.show() 
#     plt.contourf(X,Y,one_element[1][:,:,1], 50, cmap='rainbow')
#     plt.show() 







# plt.contourf(X,Y,Phas, 50, cmap='rainbow')
# plt.show()
# ax = plt.axes(projection='3d')
# plt.contourf(X,Y,Phas, 50, cmap='rainbow')
# plt.show()
# plt.plot(N,PhaR, 5)
# plt.plot(N,Amp, 5)
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

