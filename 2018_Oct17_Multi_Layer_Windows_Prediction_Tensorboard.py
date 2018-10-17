"""This code is for experimenting to build a model of predicting window opening behaviors"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def add_layer(inputs, in_size, out_size,n_layer, activation_function=None, ):
    """add one more layer and return the output of this layer"""
    layer_name="layer%s" % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name="W")
            tf.summary.histogram(layer_name+"/Weight",Weights)
        with tf.name_scope("Biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            tf.summary.histogram(layer_name+"/Biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name+"/output", outputs)
        return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


data = pd.read_csv("F:/201810_ML_Learning/Tensorflow/J1_annually_data_Handled_For_TF.csv")
print(data.columns)
x = data[['Hour', 'Month', 'Week', 'Temperature', 'RH', 'HCHO', 'CO2', 'PM2.5', 'Out_Temperature',
          'Out_RH', 'Rain', 'CO(mg/m3)', 'NO2(ug/m3)', 'SO2(ug/m3)', 'O3(ug/m3)', 'PM10(ug/m3)',
          'PM2.5(ug/m3)', 'AQI']]
y = data[['a','b']]
print(x.shape)
x_train, x_test, y_train_n, y_test_n = train_test_split(x, y, test_size=0.25)

print(x_test)


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train_n = np.array(y_train_n)
y_test_n = np.array(y_test_n)

with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 18], name='x_input')  # X_Parameters=18
    ys = tf.placeholder(tf.float32, [None, 2], name='y_input')  # Y_Parametes=2

layer1 = add_layer(xs, 18, 1000,n_layer=1, activation_function=tf.nn.sigmoid)
layer2 = add_layer(layer1, 1000, 1000,n_layer=2, activation_function=tf.nn.sigmoid)
layer3 = add_layer(layer2, 1000, 1000,n_layer=3, activation_function=tf.nn.sigmoid)
layer4 = add_layer(layer3, 1000, 1000,n_layer=4, activation_function=tf.nn.sigmoid)
layer5 = add_layer(layer4, 1000, 200, n_layer=5,activation_function=None)

# add output layer
prediction = add_layer(layer5, 200, 2, n_layer=6,activation_function=tf.nn.relu)

with tf.name_scope('loss'):
    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  # loss
    tf.summary.scalar("loss", cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("C:/Users\dengtingxuan\Desktop/logs/", sess.graph)


#print(x_test.reshape(-1,18))


# batch_xs=x_train
# batch_ys= y_train_n
# result=sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
# print(compute_accuracy(x_test.reshape(-1,18), y_test_n.reshape(-1,2)))

for i in range(len(x_train)//1000-1):
    batch_xs = x_train[i:(i+1)*1000, :].reshape(-1, 18)
    batch_ys = y_train_n[i:(i+1)*1000, :].reshape(-1, 2)
    # print(batch_xs)
    # print(batch_ys)
    result=sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    # print(compute_accuracy(x_test.reshape(-1, 18), y_test_n.reshape(-1, 2)))
    if i % 5 == 0:
         print(compute_accuracy(x_test.reshape(-1,18), y_test_n.reshape(-1,2)))
         #writer.add_summary(result,i)
         print(i)

writer.close()