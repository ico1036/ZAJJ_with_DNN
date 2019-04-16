import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

## --Read file
train_df=pd.read_csv('../DATA/train_df.csv',sep=" ")
test_df =pd.read_csv('../DATA/test_df.csv',sep=" ")

print("###### train / test Dataframe ######")
display(train_df.shape)
display(test_df.shape)


print("###### train / test Numpy ######")
train_xy = train_df.values
test_xy  = test_df.values
print(train_xy.shape,type(train_xy))
print(test_xy.shape,type(test_xy))

## --Normalize
def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator+1e-7)

train_xy = MinMaxScaler(train_xy)
test_xy  = MinMaxScaler(test_xy)



train_x_data = train_xy[:,1:]
train_y_data = train_xy[:,:1]
test_x_data = test_xy[:,1:]
test_y_data = test_xy[:,:1]


print("###### Features / Label Numpy ######")
print(train_x_data.shape)
print(train_y_data.shape)

print(train_y_data)



print("##### train data shape #####")
print(train_x_data.shape[0])

#### --Learning Algorithm



## --HyperParameter
learning_rate = 0.01
training_epochs = 20
batch_size= 100
neu=12


# --Placeholder
X = tf.placeholder(tf.float32, shape=[None,24])
Y = tf.placeholder(tf.float32, shape=[None,1])
keep_prob = tf.placeholder(tf.float32)



# --Input layer
W1 = tf.get_variable("W1", shape=[24, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b1 = tf.Variable(tf.random_normal([neu]))
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


# --Hidden layer 1
W2 = tf.get_variable("W2", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b2 = tf.Variable(tf.random_normal([neu]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# --Hidden layer 2
W3 = tf.get_variable("W3", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b3 = tf.Variable(tf.random_normal([neu]))
L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#
## --Hidden layer 3
#W4 = tf.get_variable("W4", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
#b4 = tf.Variable(tf.random_normal([neu]))
#L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
#L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
#
## --Hidden layer 4
#W5 = tf.get_variable("W5", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
#b5 = tf.Variable(tf.random_normal([neu]))
#L5 = tf.nn.relu(tf.matmul(L4,W5)+b5)
#L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
#
## --Hidden layer 5
#W6 = tf.get_variable("W6", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
#b6 = tf.Variable(tf.random_normal([neu]))
#L6 = tf.nn.relu(tf.matmul(L5,W6)+b6)
#L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

## --Hidden layer 6
#W7 = tf.get_variable("W7", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
#b7 = tf.Variable(tf.random_normal([neu]))
#L7 = tf.nn.relu(tf.matmul(L6,W7)+b7)
#L7 = tf.nn.dropout(L7, keep_prob=keep_prob)
#
## --Hidden layer 7
#W8 = tf.get_variable("W8", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
#b8 = tf.Variable(tf.random_normal([neu]))
#L8 = tf.nn.relu(tf.matmul(L7,W8)+b8)
#L8 = tf.nn.dropout(L8, keep_prob=keep_prob)

# --output layer 
W4 = tf.get_variable("W4", shape=[neu, 1],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b4 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.sigmoid(tf.matmul(L3,W4)+b4)

## --Loss and Optimizer
l2reg = 0 * tf.reduce_sum(tf.square(W3))
loss = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis)) + l2reg


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(training_epochs):
    avg_loss=0
    batch = int(train_x_data.shape[0] / batch_size)
    
    for i in range(batch):
        batch_x, batch_y = train_x_data[i*batch_size:i*batch_size+batch_size], train_y_data[i*batch_size:i*batch_size+batch_size]
        feed_dict = {X: batch_x, Y: batch_y, keep_prob:1}
    
        c, _ = sess.run([loss,optimizer],feed_dict=feed_dict )
        avg_loss += c / batch
        acc_val = sess.run(accuracy,feed_dict=feed_dict)
    print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_loss), 'acc =', '{:.2%}'.format(acc_val))
print('Learning finished')


# --Test model and check acc



a = tf.cast(tf.argmax(predicted,1),tf.float32)
b = tf.cast(tf.argmax(Y,1),tf.float32)

auc = tf.metrics.auc(b, a)
TP = tf.metrics.true_positives(b, a)
TN = tf.metrics.true_positives(b, a)
FP = tf.metrics.false_positives(b, a)
FN = tf.metrics.false_negatives(b, a)

sess.run(tf.local_variables_initializer())
acc_val = sess.run(accuracy,feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1})
print('Accuracy:','{:.2%}'.format(acc_val))
print('TP:', sess.run(TP, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1}))
print('TN:', sess.run(TN, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1}))
print('FP:', sess.run(FP, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1}))
print('FN:', sess.run(FN, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1}))
print('AUC:', sess.run(auc, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1}))

































