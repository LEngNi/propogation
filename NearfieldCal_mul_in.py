# -*- coding: utf-8 -*-
# Version: 5.1
# 大全 https://blog.csdn.net/u011746554/article/details/74393922
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
# from tensorflow.keras import backend as K
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

DataPath,LogPath,TrainImgPath = '.\\Images_mul_in', '.\\Images_mul_in', '.\\Images_mul_in'
if not os.path.isdir(DataPath): os.makedirs(DataPath)
if not os.path.isdir(LogPath):  os.makedirs(LogPath)
if not os.path.isdir(TrainImgPath): os.makedirs(TrainImgPath)  


Xrange,Yrange=128,128
X, Y = np.linspace(0, Xrange-1,Xrange,dtype=int), np.linspace(0, Yrange-1,Yrange,dtype=int)

#--------------------------------------------
#定义网络结构 Functional API方式
inputs = tf.keras.Input(shape=(Xrange, Yrange , 2)) #需要（长，宽，通道数）
inputs_para = tf.keras.Input(shape = (3,))
#inputs_pad = tf.keras.layers.ZeroPadding2D(((14,13),(14,13)))(inputs)
#print(inputs_pad.shape)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
#c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
#c1 = tf.keras.layers.BatchNormalization()(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
#c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
#c2 = tf.keras.layers.BatchNormalization()(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
#c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
c3 = tf.keras.layers.BatchNormalization()(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
#c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
#c4 = tf.keras.layers.BatchNormalization()(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
#c5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
#c5 = tf.keras.layers.BatchNormalization()(c5)

c1_para = tf.keras.layers.Dense(16, activation = 'relu')(inputs_para)
c2_para = tf.keras.layers.Dense(64, activation ='relu')(c1_para)
c3_para = tf.keras.layers.Reshape((8,8,1))(c2_para)

combine = tf.keras.layers.concatenate([c5,c3_para])


u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(combine)
#u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
#c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
#c6 = tf.keras.layers.BatchNormalization()(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
#c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
#c7 = tf.keras.layers.BatchNormalization()(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
#c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
#c8 = tf.keras.layers.BatchNormalization()(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
#c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
#c9 = tf.keras.layers.BatchNormalization()(c9)

outputs = tf.keras.layers.Conv2D(2, (1, 1))(c9)
model = tf.keras.Model(inputs=[inputs,inputs_para], outputs=[outputs])

#自定义loss函数
"""
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
"""
# from keras import backend as K
# def mean_iou(y_true, y_pred):  #？？？？
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.cast(y_pred > t, tf.int32)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
  #  optimizer='sgd', # SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loss =[tf.keras.losses.MeanSquaredError()], #引用自定义loss 固定输入输出接口
    metrics=[tf.keras.metrics.MeanSquaredError()]# metrics=[mean_iou]??
)

print(model.summary()) #打印神经网络结构，统计参数数目

#--------------------------------------------
# 读取 TFRecord 文件
tfrecord_file = '.\\Images_mul_in\\TrainData.tfrecords'
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)   
#读取验证集文件
# tfrecord_file_val = '.\\Images\\ValidationData.tfrecords'
# raw_dataset_val = tf.data.TFRecordDataset(tfrecord_file_val)


feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'Source': tf.io.FixedLenFeature([Xrange,Yrange,2], tf.float32),
    'Para': tf.io.FixedLenFeature([3,], tf.float32),
    'Target': tf.io.FixedLenFeature([Xrange,Yrange,2], tf.float32) }

def read_example(example_string): #    从TFrecord格式文件中读取数据
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    return ((feature_dict['Source'],feature_dict['Para'])), feature_dict['Target']



NumSamples=1000
buffer_size=NumSamples  #总数
batch_size=16    #每次训练样本数，平均后产生梯度即可
#num_batches=buffer_size//batch_size #样本批次
EPOCHS=20 #重复一整套训练数据的次数

train_dataset = raw_dataset.map(read_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)#.repeat(5) 找199个出来，设每批次399/40个

# val_dataset = raw_dataset_val.map(read_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)

# #--------------------------------------------
# # 从文件恢复模型参数,暂时根据需要手动控制

checkpoint = tf.train.Checkpoint(Mymodel=model)
manager = tf.train.CheckpointManager(checkpoint, LogPath, max_to_keep=2)
#checkpoint.restore(manager.latest_checkpoint)
print("Checkpoint loaded:\n",checkpoint)


#建立监控
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LogPath, histogram_freq=1)
train_summary_writer = tf.summary.create_file_writer(LogPath)

#早停策略
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='MSE', min_delta=0.01,patience=3)




#训练 # validation_data 可以再用测试数据。打乱，分批已经处理过,这是是重复，插入监控
history = model.fit(train_dataset, epochs=EPOCHS)#,validation_data = val_dataset)#,validation_data=train_dataset)

#记录保存模型参数，待后续使用
#manager.save()
print("checkpoint saved.")

# https://tensorflow.google.cn/guide/keras/save_and_serialize?hl=en
model.save(LogPath+"\\ModelSaved_mul_in")#,custom_objects={"dice_loss": dice_loss}) # 或者 tf.keras.models.save_model(model,LogPath+"\ModelSaved") 
print('Model saved.') # 最后默认生成 .pb 格式模型
#model=tf.keras.models.load_model(LogPath+"\\ModelSaved")
#print('Model Loaded')

#预测一次阵列激励，方向图np输入 可以不编译compile 返回值是预测值的numpy array
def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


FieldSource=np.load(DataPath+'\\validation_data\\Source\\source1.npy')   
FieldTarget=np.load(DataPath+'\\validation_data\\Target\\target1.npy')  
FieldPara = np.zeros(3,dtype = np.float32)
FieldPara[0] = FieldSource[0,0,2]
FieldPara[1] = FieldSource[0,0,3]
FieldPara[2] = FieldSource[0,0,4]
Field = np.zeros([Xrange,Yrange,2],dtype = np.float32)
Field[:,:,0] =FieldSource[:,:,0]
Field[:,:,1] = FieldSource[:,:,1]
Field = Field.reshape(-1,Xrange,Yrange,2)
FieldPara = FieldPara.reshape(-1,3)
FieldInput = [Field,FieldPara] 
#Mask=np.load(DataPath+'\\validation_data\\Mask\\mask0.npy')  
#FieldSource=FieldSource.reshape(-1,Xrange,Yrange,5)
FieldPredic=model.predict(FieldInput) 

FieldPredicAmp = FieldPredic[:,:,:,0].reshape(Xrange,Yrange)
FieldPredicPha = FieldPredic[:,:,:,1].reshape(Xrange,Yrange)
FieldTargetAmp = FieldTarget[:,:,0]
FieldTargetPha = FieldTarget[:,:,1]
plt.contourf(X,Y,FieldTargetAmp, 50, cmap='rainbow')
plt.show() 
plt.contourf(X,Y,FieldTargetPha, 50, cmap='rainbow')
plt.show() 
plt.contourf(X,Y,FieldPredicAmp, 50, cmap='rainbow')
plt.show() 
plt.contourf(X,Y,FieldPredicPha, 50, cmap='rainbow')
plt.show() 

'''
import pickle
 
with open('trainHistoryDict.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
with open('trainHistoryDict.txt','rb') as file_pi:
    history=pickle.load(file_pi)
    
'''
#---------------------------------------------------
# 2-使用tf.GradientTape()求解梯度，这样可以自定义训练过程。
# https://cloud.tencent.com/developer/article/1552598
# 好像如果不需要监控batch的loss，也可以返回fit
#网络的训练部分
# EPOCHS=300 #重复一整套训练数据的次数
# batch_size=1    #每次训练样本数，平均后产生梯度即可
# for epoch in range(EPOCHS):
#     with tf.GradientTape() as tape:
#           predictions = model(image, training=True) #获得一批predictions,优化所以选择批1次
#           Myloss = MyLossFunc2(image, predictions)  #获得一批数据得到的单一Myloss
#     train_loss(Myloss)
#     gradients = tape.gradient(Myloss, model.trainable_variables)
# #    optimizer.minimize(gradients,model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     # with train_summary_writer.as_default():  #网页监控用
#     #      tf.summary.scalar('loss', train_loss.result(), step=epoch)
#     print('Epoch:',epoch,batch_size,'Loss:',train_loss.result().numpy())#,'Accuracy:',train_accuracy.result().numpy())
#     train_loss.reset_states() # Reset metrics every epoch

#---------------------------------------------------