import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tqdm.notebook import tqdm
from tcn import TCN
plt.style.use('seaborn')
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import random
random_seed = 1388
random.seed(random_seed)
np.random.seed(random_seed)

import tensorflow as tf

from tensorflow import set_random_seed
set_random_seed(2)

np.set_printoptions(suppress=True)
#设置神经网络参数
batch_size=64#批训练大小
epoch=50#迭代次数
test_ratio=.2#测试集比例
windows=7#时间窗
scale=1.0#归一化参数
# adjust_param = 0.02 #模型调节值


x_train = np.load("File/dataFile/X_train.npy", allow_pickle=True)
y_train = np.load("File/dataFile/y_train.npy", allow_pickle=True)
x_test = np.load("File/dataFile/X_test.npy", allow_pickle=True)
y_test = np.load("File/dataFile/y_test.npy", allow_pickle=True)
amount_of_features = x_train.shape[1]

def mape(y_true, y_pred):
  #评价指标MAPE
  record=[]
  for index in range(len(y_true)):
    if abs(y_true[index])>10:
      temp_mape=np.abs((y_pred[index] - y_true[index]) / y_true[index])
      record.append(temp_mape)
  return np.mean(record) * 100


def easy_result(y_train,y_train_predict,train_index):
  #进行反归一化

  X_scaler = load(open('File/dataFile/X_scaler.pkl', 'rb'))
  y_scaler = load(open('File/dataFile/y_scaler.pkl', 'rb'))
  train_predict_index = np.load("File/dataFile/index_train.npy", allow_pickle=True)
  test_predict_index = np.load("File/dataFile/index_test.npy", allow_pickle=True)

  # y_train_predict=y_train_predict[:,0]

  y_train = y_scaler.inverse_transform(y_train)
  y_train_predict = y_scaler.inverse_transform(y_train_predict)

   #画图进行展示
  plt.figure(figsize=(10,5))
  plt.plot(y_train[:],color='#ff5b00')
  plt.plot(y_train_predict[:],color='blue')
  plt.legend(('real', 'predict'),fontsize='15')
  plt.title("%s Data"%train_index,fontsize='20') #添加标题
  plt.show()
  print('\n')
  plot_begin,plot_end=min(min(y_train),min(y_train_predict)),max(max(y_train),max(y_train_predict))
  plot_x=np.linspace(plot_begin,plot_end,10)
  plt.figure(figsize=(5,5))
  plt.plot(plot_x,plot_x)
  plt.plot(y_train,y_train_predict,'o')
  plt.title("%s Data"%train_index,fontsize='20') #添加标题
  plt.show()
  #输出结果
  print('%s上的MAE/MSE/RMSE/MAPE/R^2'%train_index)
  print(mean_absolute_error(y_train, y_train_predict))
  print(mean_squared_error(y_train, y_train_predict))
  print(np.sqrt(mean_squared_error(y_train, y_train_predict) ))
  print(mape(y_train, y_train_predict) )
  print(r2_score(y_train, y_train_predict))

def TNet_attention_model():
    inputs = Input(shape=(amount_of_features,windows))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    tcn = TCN(nb_filters=amount_of_features)(x)
    attention=Dense(amount_of_features, activation='sigmoid', name='attention_vec')(tcn)#求解Attention权重
    attention=Activation('softmax',name='attention_weight')(attention)
    tcn=Multiply()([tcn, attention])#attention与tcn对应数值相乘
    outputs = Dense(1, activation='linear')(tcn)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile('adam','mae')
    model.summary()
    return model

TNet_Model = TNet_attention_model()

history = TNet_Model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=50,#个股
                      verbose=2,
                      validation_split=0.1)

#迭代图像
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))
plt.plot(epochs_range, loss, label='A-TrellisNet Loss')
plt.plot(epochs_range, val_loss, label='A-TrellisNet Loss')
plt.legend(loc='upper right')
plt.title('A-TrellisNet Train and Test Loss')
plt.show()

#指标数据
y_train_predict=TNet_Model.predict(x_train)#预测结果
easy_result(y_train,y_train_predict,'A-TrellisNet Train')#输出评价指标
y_test_predict=TNet_Model.predict(x_test)#预测结果
easy_result(y_test,y_test_predict,'002594')#输出评价指标