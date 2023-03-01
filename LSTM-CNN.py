import tensorflow as tf
import numpy as np
import collections
import codecs
import jieba
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt

datapaths = r"LSTM&CNN/data/"

positive_data = []
y_positive = []
neutral_data = []
y_neutral = []
negative_data = []
y_negative = []

print("#------------------------------------------------------#")
print("加载数据集")
with codecs.open(datapaths + "pos.csv", "r", "utf-8") as f1, \
        codecs.open(datapaths + "neutral.csv", "r", "utf-8") as f2, \
        codecs.open(datapaths + "neg.csv", "r", "utf-8") as f3:
    for line in f1:
        positive_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
        y_positive.append([1, 0, 0])
    for line in f2:
        neutral_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
        y_neutral.append([0, 1, 0])
    for line in f3:
        negative_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
        y_negative.append([0, 0, 1])

print("positive data:{}".format(len(positive_data)))
print("neutral data:{}".format(len(neutral_data)))
print("negative data:{}".format(len(negative_data)))

x_text = positive_data + neutral_data + negative_data
y_label = y_positive + y_neutral + y_negative
print("#------------------------------------------------------#")
print("\n")

max_document_length=200
min_frequency=1

vocab = "";
x = np.array(list(vocab.fit_transform(x_text)))
vocab_dict = collections.OrderedDict(vocab.vocabulary_._mapping)

print("#----------------------------------------------------------#")
print("数据混洗")
np.random.seed(10)
y = np.array(y_label)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

test_sample_percentage = 0.2
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

train_positive_label = 0
train_neutral_label = 0
train_negative_label = 0
test_positive_label = 0
test_neutral_label = 0
test_negative_label = 0

for i in range(len(y_train)):
    if y_train[i, 0] == 1:
        train_positive_label += 1
    elif y_train[i, 1] == 1:
        train_neutral_label += 1
    else:
        train_negative_label += 1

for i in range(len(y_test)):
    if y_test[i, 0] == 1:
        test_positive_label += 1
    elif y_test[i, 1] == 1:
        test_neutral_label += 1
    else:
        test_negative_label += 1

print("训练集中 positive 样本个数：{}".format(train_positive_label))
print("训练集中 neutral 样本个数：{}".format(train_neutral_label))
print("训练集中 negative 样本个数：{}".format(train_negative_label))
print("测试集中 positive 样本个数：{}".format(test_positive_label))
print("测试集中 neutral 样本个数：{}".format(test_neutral_label))
print("测试集中 negative 样本个数：{}".format(test_negative_label))

print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("读取預训练词向量矩阵")

pretrainpath = r"LSTM&CNN/MODEL/"

embedding_index = {}

not_use_code = ['=', '"', '+', '−', '→', '...,', '∈', '≤', '...', '≥', '记', '歌', '(', ')']

with codecs.open(pretrainpath + "sgns.wiki.bigram-char", "r", "utf-8") as f:
    line = f.readline()
    nwords = int(line.strip().split(" ")[0])
    ndims = int(line.strip().split(" ")[1])
    for line in f:
        values = line.split()
        try:
            words = values[0]
            values_use = [i for i in values[1:] if i not in not_use_code]
            coefs = np.asarray(values_use, dtype="float32")
            embedding_index[words] = coefs
        except IndexError:
            pass
        except:
            pass

print("預训练模型中Token总数：{} = {}".format(nwords, len(embedding_index)))
print("預训练模型的维度：{}".format(ndims))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("将vocabulary中的 index-word 对应关系映射到 index-word vector形式")

embedding_matrix = []
notfoundword = 0


batch_size=500
max_sentence_length=200
lstm_output_size=128
embedding_dims=ndims
filters = 250
kernel_size = 3
dropout=0.2
recurrent_dropout=0.2
num_classes=3
epochs=15

# 定义网络结构 LSTM+CNN
model_two=Sequential()
model_two.add(Embedding(128,
                    embedding_dims,
                    weights=[embedding_matrix],
                    input_length=max_sentence_length,
                    trainable=False))
model_two.add(Dropout(dropout))
model_two.add(LSTM(lstm_output_size))
model_two.add(Activation("sigmoid"))
model_two.add(Dense(num_classes))
model_two.add(Conv1D(filters,kernel_size,padding="valid",activation="relu",strides=1))
model_two.add(MaxPooling1D())
model_two.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history_two = model_two.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))

plt.figure()

plt.plot(epochs,history_two.history['acc'],'b',label='accuracy in LSTM')
plt.title('Accuracy In LSTM And LSTM-CNN')
plt.legend()
plt.savefig('LSTM&CNN/figure/model_acc.jpg')

plt.figure()

plt.plot(epochs,history_two.history['loss'],'b',label='loss in LSTM')
plt.title('Loss In LSTM And LSTM-CNN')
plt.legend()
plt.savefig('LSTM&CNN/figure/model_loss.jpg')

plt.figure()

plt.plot(epochs,history_two.history['val_loss'],'r',label='val_loss in LSTM-CNN')
plt.title('val_loss In LSTM And LSTM-CNN')
plt.legend()
plt.savefig('LSTM&CNN/figure/model_val_loss.jpg')

plt.figure()

plt.plot(epochs,history_two.history['val_acc'],'r',label='val_acc in LSTM-CNN')
plt.title('val_acc In LSTM And LSTM-CNN')
plt.legend()
plt.savefig('LSTM&CNN/figure/model_val_acc.jpg')