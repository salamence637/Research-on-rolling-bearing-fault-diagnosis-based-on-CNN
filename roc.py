
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
epochs = 12
num_classes = 10
length = 2048
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.6, 0.2, 0.2]  # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path, length=length,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=True, enc_step=28)

x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]
# 输入数据的维度
input_shape = x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

mark = time.strftime("%Y%m&d_%H%M", time.localtime())
model_name = "{}-rounds-{}".format(1, mark)
tb_cb = TensorBoard(log_dir='logs\{}'.format(model_name))
print(model_name)
model = Sequential()
# 第一层卷积
model.add(
    Conv1D(filters=16, kernel_size=7, strides=7, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4, padding='valid'))

# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层，共100个单元，激活函数为ReLU
model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))

model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
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



y_pred = model.predict(x_test,batch_size=batch_size)
#Y_train0为真实标签，Y_pred_0为预测标签，注意，这里roc_curve为一维的输入，Y_train0是一维的
fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
auc = auc(fpr, tpr)
print("AUC : ", auc)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.savefig('A_ROC/0123-4val.jpg')
plt.show()
