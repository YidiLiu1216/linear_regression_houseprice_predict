import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#readin data
house_info=pd.read_csv('house.csv')
#print (house_info.describe())

house_info=house_info.values
house_info=np.array(house_info)

#data prepare
tmpx,tmpy=house_info[:,:12],house_info[:,12]
#prameter normalization
for i in range (12):
    tmpx[:,i]=(tmpx[:,i]-tmpx[:,i].mean())/(tmpx[:,i].max()-tmpx[:,i].min())
train_x,test_x,train_y,test_y=train_test_split(tmpx,tmpy,test_size=0.1)

#define model
x = tf.placeholder(tf.float32, [None,12], name = "X")
y = tf.placeholder(tf.float32, [None,1], name = "Y")

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name="W")
    b = tf.Variable(1.0, name="b")

    def model(x, w, b):
        return tf.matmul(x, w) + b
    pred = model(x, w, b)

#testcode
def test():
  acc=0.0
  for xs, ys in zip(test_x, test_y):
      xs = xs.reshape(1, 12)
      ys = ys.reshape(1, 1)
      predict=sess.run(pred,feed_dict={x:xs})
      acc+=(1-abs(predict-ys[0])/ys[0])
  acc/=len(test_y)
  #draw result

  print(acc)
  return acc[0][0]

#train&test
train_epoch=25
learning_rate = 0.003
with tf.name_scope("LostFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_list=[]
acc_list=[]
for epoch in range(train_epoch):
    loss_sum = 0.0

    for xs,ys in zip(train_x,train_y):

        xs = xs.reshape(1, 12)
        ys = ys.reshape(1, 1)

        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})
        loss_sum = loss_sum + loss
    acc_list.append(test())
    train_x, train_y = shuffle(train_x, train_y)
    b_temp = b.eval(session=sess)
    w_temp = w.eval(session=sess)

    loss_average = loss_sum / len(train_y)
    loss_list.append(loss_average)


sess.close()

plt.subplot(1,2,1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss_list)
plt.subplot(1,2,2)
plt.xlabel('epoch')
plt.ylabel('accuary')
plt.plot(acc_list)
plt.show()
#test

