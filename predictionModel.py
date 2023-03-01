import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tqdm.notebook import tqdm
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
import time
random_seed = 1388
random.seed(random_seed)
np.random.seed(random_seed)

import tensorflow as tf

from trellisnet import TrellisNet

from tensorflow import set_random_seed
set_random_seed(2)

np.set_printoptions(suppress=True)
#设置神经网络参数
batch_size=128#批训练大小
epoch=70#迭代次数
test_ratio=.2#测试集比例
windows=7#时间窗
scale=1.0#归一化参数
# adjust_param = 0.02 #模型调节值

#读取数据
# data = pd.read_csv('N225.csv').iloc[:,1:]
data = pd.read_csv(r'D:\多因子实验\英国数据\英国数据\wpp.csv').iloc[::-1]

data.replace(0, 0+1e-6, inplace=True)

data.replace(np.nan, 0+1e-6, inplace=True)

data = data.drop(['volume'], axis = 1)
data = data.drop(['instrument'], axis = 1)
# data = data.drop(['bdIndex'], axis = 1)
data = data.drop(['datetime'], axis = 1)
data = data.sort_index(ascending=True)
# print(data.tail())

y=data['close']
data = np.array(data)/scale
cut = round(test_ratio* data.shape[0])
amount_of_features=data.shape[1]
lstm_input=[]
data_temp=data
for i in range(len(data_temp)-windows):
    lstm_input.append(data_temp[i:i+windows,:])

lstm_input=np.array(lstm_input)


lstm_output=y[:-windows]

lstm_output=np.array(lstm_output)

x_train,y_train,x_test,y_test=\
lstm_input[:-cut,:,:],lstm_output[:-cut:],lstm_input[-cut:,:,:],lstm_output[-cut:]
# print(x_train)
# print("-----------------------------")
# print(y_train)
# print("-----------------------------")
# print(x_test)
# print("-----------------------------")
# print(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

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
  #y_train_predict=np.reshape(y_train_predict, (-1,1))
  #y_train_predict= mm_y.inverse_transform(y_train_predict)
  y_train_predict=y_train_predict[:,0]
  #y_train=np.reshape(y_train, (-1,1))
  #y_train=mm_y.inverse_transform(y_train)
  #y_train=y_train[:,0]
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
    inputs = Input(shape=(windows,amount_of_features))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    tre = TrellisNet(nb_filters=amount_of_features)(x)
    attention=Dense(amount_of_features, activation='sigmoid', name='attention_vec')(tre)#求解Attention权重
    attention=Activation('softmax',name='attention_weight')(attention)
    tcn=Multiply()([tre, attention])#attention与tcn对应数值相乘
    outputs = Dense(1, activation='linear')(tcn)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile('adam','mae')
    model.summary()
    return model

TNet_Model = TNet_attention_model()

print("开始时间：")
start = time.perf_counter()
history = TNet_Model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=30,#个股
                      verbose=2,
                      validation_split=0.1)

print("结束时间：")
end = time.perf_counter()

print('用时：{:.4f}s'.format(end-start))

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