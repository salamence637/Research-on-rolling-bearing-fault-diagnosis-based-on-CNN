
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2
import preprocess
import time
from keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from keras.optimizers import Adam,SGD,sgd
from keras.models import load_model



# 训练参数
batch_size = 128
epochs = 200
num_classes = 10
# length = 2048
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.8, 0.1, 0.1]  # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=False)

# x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]
# 输入数据的维度
input_shape = x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

mark = time.strftime("%Y%m%d_%H%M", time.localtime())
model_name = "{}-rounds-{}".format(1, mark)
tb_cb = TensorBoard(log_dir='logs\{}'.format(model_name))
print(model_name)
model = Sequential()

# 第一层nn

model.add(Dense(units=100, input_dim=864, activation='sigmoid'))
model.add(Dense(units=50, input_dim=100, activation='sigmoid'))
# model.add(Dense(units=100, input_dim=500, activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))


# 编译模型
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard调用查看一下训练情况
tb_cb = TensorBoard(log_dir='logs\{}'.format(model_name))

# 开始模型训练
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
          callbacks=[tb_cb])

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
plot_model(model=model, to_file='cnn-1D.png', show_shapes=True)

# y_pred = model.predict(x_test,batch_size=batch_size)
# #Y_train0为真实标签，Y_pred_0为预测标签，注意，这里roc_curve为一维的输入，Y_train0是一维的
# fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
# auc = auc(fpr, tpr)
# print("AUC : ", auc)
# plt.figure()
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# #plt.savefig('A_ROC/0123-4val.jpg')
# plt.show()
