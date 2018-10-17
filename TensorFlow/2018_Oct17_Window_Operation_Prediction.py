"""This code is for experimenting to build a model of predicting window opening behaviors"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def add_layer(inputs, in_size, out_size, activation_function=None ,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1 ,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b ,)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre ,1), tf.argmax(v_ys ,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
data=pd.read_csv("F:/201810_ML_Learning/J1_annually_data_Handled_For_TF.csv")
print(data.columns)
x=data[['Hour', 'Month', 'Week', 'Temperature', 'RH', 'HCHO', 'CO2', 'PM2.5','Out_Temperature',
        'Out_RH', 'Rain', 'CO(mg/m3)', 'NO2(ug/m3)','SO2(ug/m3)', 'O3(ug/m3)', 'PM10(ug/m3)',
        'PM2.5(ug/m3)', 'AQI']]
y=data['BRW']
print(x.shape)
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.25)
print(y_train)
y_train_n=pd.DataFrame()
y_test_n=pd.DataFrame()
a0=pd.DataFrame({"a":[1],"b":[0]},columns=["a","b"])
a1=pd.DataFrame({"a": [0], "b": [1]}, columns=["a", "b"])

for i in range(len(y_train)):
    if y_train.iloc[i]==1:
        y_train_n=y_train_n.append(a0)
    else:
        y_train_n =y_train_n.append(a1)
    print(y_train_n)

for i in range(len(y_test)):
    if y_test.iloc[i]==1:
        y_test_n=y_test_n.append(a0)
    else:
        y_test_n =y_test_n.append(a1)
    print(y_test_n)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 18]) # X_Parameters=18
ys = tf.placeholder(tf.float32, [None, 2])  # Y_Parametes=2

# add output layer
prediction = add_layer(xs, 18, 2,  activation_function=tf.nn.relu)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())



x_train=np.array(x_train)
x_test=np.array(x_test)
y_train_n=np.array(y_train_n)
y_test_n=np.array(y_test_n)



for i in range(0,10000,100):
    batch_xs= x_train[i:i+100,:].reshape(-1,18)
    batch_ys = y_train_n[i:i+100,:].reshape(-1,2)
    # print(batch_xs)
    # print(batch_ys)

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    print(i)
    if i % 50 == 0:
         print(compute_accuracy(x_test.reshape(-1,18), y_test_n.reshape(-1,2)))
        # print(batch_xs, batch_ys)


# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
#     if i % 50 == 0:
#         print(compute_accuracy(
#             mnist.test.images, mnist.test.labels))