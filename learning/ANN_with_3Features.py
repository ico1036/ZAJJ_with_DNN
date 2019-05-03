import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pandas import Series, DataFrame
import math
from sklearn.utils import shuffle


## --Physics varaibles

Lumi=150000

N_gen_sig = 11994591.0
N_gen_bkg = 13999751.0
N_gen_tt = 1000000
N_gen_zz = 1000000
N_gen_za = 8000000
N_gen_zaj = 1000000
N_gen_zajj120M = 999930
N_gen_zajj600M = 999986
N_gen_zajj1000M = 999835

N_pre_sel_sig = 1264236.0
N_pre_sel_bkg = 153504.0
N_pre_sel_tt = 318
N_pre_sel_zz = 14
N_pre_sel_za = 8204
N_pre_sel_zaj = 10265
N_pre_sel_zajj120M = 35228
N_pre_sel_zajj600M = 48402
N_pre_sel_zajj1000M = 51073

xsec_za          =7.917
xsec_zaj         =2.722
xsec_zajjQCD120  = 0.5274
xsec_zajjQCD600  = 0.02727
xsec_zajjQCD1000 = 0.008706
xsec_ZZ          = 9.339
xsec_tt          = 2.074
xsec_sig         = 0.01291

eff_za          = Lumi * xsec_za / N_gen_za     
eff_zaj         = Lumi * xsec_zaj / N_gen_zaj
eff_zajjQCD120  = Lumi * xsec_zajjQCD120 / N_gen_zajj120M
eff_zajjQCD600  = Lumi * xsec_zajjQCD600 / N_gen_zajj600M
eff_zajjQCD1000 = Lumi * xsec_zajjQCD1000 / N_pre_sel_zajj1000M
eff_ZZ          = Lumi * xsec_ZZ / N_gen_zz
eff_tt          = Lumi * xsec_tt / N_gen_tt
eff_sig         = Lumi * xsec_sig / N_gen_sig

# Method 2

N_expected_signal = Lumi * xsec_sig * N_pre_sel_sig / N_gen_sig
N_expected_tt = Lumi * xsec_tt * N_pre_sel_tt / N_gen_tt
N_expected_zz = Lumi * xsec_ZZ * N_pre_sel_zz / N_gen_zz
N_expected_za = Lumi * xsec_za * N_pre_sel_za / N_gen_za
N_expected_zaj = Lumi * xsec_zaj * N_pre_sel_zaj / N_gen_zaj
N_expected_zajj120M = Lumi * xsec_zajjQCD120 * N_pre_sel_zajj120M / N_gen_zajj120M 
N_expected_zajj600M = Lumi * xsec_zajjQCD600 * N_pre_sel_zajj600M / N_gen_zajj600M 
N_expected_zajj1000M = Lumi * xsec_zajjQCD1000 * N_pre_sel_zajj1000M / N_gen_zajj1000M
N_expected_bkg = N_expected_tt+N_expected_zz+N_expected_za+N_expected_zaj+N_expected_zajj120M+N_expected_zajj600M+N_expected_zajj1000M

N_signal_sorting = int(N_expected_signal * 46052 / N_expected_bkg)

## --Read file
train_df=pd.read_csv('../DATA/train_all_df.csv',sep=" ")
test_df =pd.read_csv('../DATA/test_all_df.csv',sep=" ")
#rest_sig_df =pd.read_csv('../DATA/Rest_Signal.csv',sep=" ")

## --Make data frame consist of 3 features
train_df_copy = DataFrame(columns=("nttype","ntjjM", "ntjdEta", "ntZpVar","ntxsec","ntgenN"))
test_df_copy  = DataFrame(columns=("nttype","ntjjM", "ntjdEta", "ntZpVar","ntxsec","ntgenN"))
#rest_sig_df_copy  = DataFrame(columns=("nttype","ntjjM", "ntjdEta", "ntZpVar"))


train_df_copy["nttype"]=train_df.nttype
train_df_copy["ntjjM"]=train_df.ntjjM
train_df_copy["ntjdEta"]=train_df.ntjdEta
train_df_copy["ntZpVar"]=train_df.ntZpVar
train_df_copy["ntxsec"]=train_df.ntxsec
train_df_copy["ntgenN"]=train_df.ntgenN

test_df_copy["nttype"]  =test_df.nttype
test_df_copy["ntjjM"]  =test_df.ntjjM
test_df_copy["ntjdEta"]=test_df.ntjdEta
test_df_copy["ntZpVar"]=test_df.ntZpVar
test_df_copy["ntxsec"]=test_df.ntxsec
test_df_copy["ntgenN"]=test_df.ntgenN

# Add rest signal data
#rest_sig_df_copy["nttype"]  =rest_sig_df.nttype
#rest_sig_df_copy["ntjjM"]   =rest_sig_df.ntjjM
#rest_sig_df_copy["ntjdEta"] =rest_sig_df.ntjdEta
#rest_sig_df_copy["ntZpVar"] =rest_sig_df.ntZpVar
#test_df_copy=pd.concat([test_df_copy,rest_sig_df_copy])
#from sklearn.utils import shuffle
#test_df_copy =shuffle(test_df_copy)
#print( "### rest signal added ###")

print(" #################### tran / test dataframe ###########################")
display(train_df_copy.shape)
display(test_df_copy.shape)


################################### --Method2 division of test data s,b ratio
test_signal_df = test_df_copy[test_df_copy['nttype'] == 1]
#test_signal_df = test_signal_df.sample(n=int(N_expected_signal))
test_bkg_df    = test_df_copy[test_df_copy['nttype'] == 0]
#test_bkg_df = test_bkg_df.sample(n=int(N_expected_bkg))

print(" #################### test data: method2 ###########################")
print(" ######## Additional Sampling is applied on  test_signal ###########")

print("test signal: ",test_signal_df.shape)
print("test bkg   : ",test_bkg_df.shape)
test_df_copy = pd.concat([test_signal_df,test_bkg_df])
test_df_copy = shuffle(test_df_copy)
print("test data : ",test_df_copy.shape)
print("train data: ",train_df_copy.shape)

## --check the number 
test_signal_num,_ = test_df_copy[test_df_copy['nttype'] == 1].shape
test_bkg_num,_    = test_df_copy[test_df_copy['nttype'] == 0].shape

## --Numpy array
print(" #################### train / test Numpy  ###########################")
train_xy = train_df_copy.values
test_xy  = test_df_copy.values
print(train_xy.shape,type(train_xy))
print(test_xy.shape,type(test_xy))



## --Normalize
def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator+1e-7)
    #return numerator / (denominator)



#--origin
test_xy_origin  = test_xy
test_x_data_origin = test_xy_origin[:,1:]
test_y_data_origin = np.rint(test_xy_origin[:,:1])

#--Normalize for training
train_xy = MinMaxScaler(train_xy)
test_xy  = MinMaxScaler(test_xy)
train_x_data = train_xy[:,1:4]
train_y_data = np.rint(train_xy[:,:1])
test_x_data = test_xy[:,1:4]
test_y_data = np.rint(test_xy[:,:1])

print(" ################## Feautre / Label in trainset  ######################")
print(train_x_data.shape)
print(train_y_data.shape)
#display(train_y_data)


############## --Training


## --HyperParameter
learning_rate = 0.005
training_epochs = 7
batch_size= 100
neu=100
lammda = 0

# --Placeholder
X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])
keep_prob = tf.placeholder(tf.float32)

# --Input layer
W1 = tf.get_variable("W1", shape=[3, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
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

# --Hidden layer 3
W4 = tf.get_variable("W4", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b4 = tf.Variable(tf.random_normal([neu]))
L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


# --Hidden layer 3
W5 = tf.get_variable("W5", shape=[neu, neu],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b5 = tf.Variable(tf.random_normal([neu]))
L5 = tf.nn.relu(tf.matmul(L4,W5)+b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

# --output layer 
W6 = tf.get_variable("W6", shape=[neu, 1],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b6 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.sigmoid(tf.matmul(L5,W6)+b6)

l2reg = lammda * tf.reduce_sum(tf.square(W5))


# --Optimizer,Loss,Acc
loss = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis)) + l2reg
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
correct = tf.equal(predicted, Y)
issig=tf.equal(Y,1)



## --Learing
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learing started')
for epoch in range(training_epochs):
    avg_loss=0
    #batch = int(train_x_data.shape[0] / batch_size)
    batch = int(math.ceil(train_x_data.shape[0] / batch_size))
    
    for i in range(batch):
        batch_x, batch_y = train_x_data[i*batch_size:i*batch_size+batch_size], train_y_data[i*batch_size:i*batch_size+batch_size]
        feed_dict = {X: batch_x, Y: batch_y, keep_prob:1}
    
        c, _ = sess.run([loss,optimizer],feed_dict=feed_dict )
        avg_loss += c / batch
        acc_val = sess.run(accuracy,feed_dict=feed_dict)
    print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_loss), 'acc =', '{:.2%}'.format(acc_val))
print('Learning finished')


## --Test

a = tf.cast(tf.argmax(predicted,1),tf.float32)
b = tf.cast(tf.argmax(Y,1),tf.float32)

_,auc = tf.metrics.auc(Y, predicted)
_,TP = tf.metrics.true_positives(Y, predicted)
_,TN = tf.metrics.true_negatives(Y, predicted)
_,FP = tf.metrics.false_positives(Y, predicted)
_,FN = tf.metrics.false_negatives(Y, predicted)

sess.run(tf.local_variables_initializer())
prediction,acc_val,corr_val,issig_val = sess.run([hypothesis,accuracy,correct,issig],feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1})

########### --ROC unit

tp=sess.run(TP, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1})
tn=sess.run(TN, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1})
fp=sess.run(FP, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1})
fn=sess.run(FN, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1})


corr_val=corr_val.flatten()
not_corr_val=np.invert(corr_val)

issig_val = issig_val.flatten()
isbkg_val = np.invert(issig_val)

tp_bool = corr_val * issig_val
tn_bool = corr_val * isbkg_val
fn_bool = not_corr_val * issig_val
fp_bool = not_corr_val * isbkg_val

pre_sig_bool = tp_bool + fp_bool
pre_bkg_bool = fn_bool + tn_bool


#print(len(x_val[:,1]))
#print(len(x_val[corr_val][:,1]))

########### --Histogram

hist_obj=0 # Mjj
#hist_obj=1 # Djjeta
#hist_obj=2 # Zeppen feld


## Extraci index of each samples

hist_sig  = test_x_data_origin[issig_val]
hist_bkg  = test_x_data_origin[isbkg_val]
hist_tp = test_x_data_origin[tp_bool]
hist_fn = test_x_data_origin[fn_bool]
hist_tn = test_x_data_origin[tn_bool]
hist_fp = test_x_data_origin[fp_bool]
hist_pre_sig = test_x_data_origin[pre_sig_bool]
hist_pre_bkg = test_x_data_origin[pre_bkg_bool]

hist_hypo_sig = prediction[issig_val]
hist_hypo_bkg = prediction[isbkg_val]


print("##### sig ####")
print(hist_sig)
print("##### bkg ####")
print(hist_bkg)

hist_sig_obj  = hist_sig[:,hist_obj]
hist_bkg_obj  = hist_bkg[:,hist_obj]
hist_tp_obj = hist_tp[:,hist_obj]
hist_fn_obj = hist_fn[:,hist_obj]
hist_tn_obj = hist_tn[:,hist_obj]
hist_fp_obj = hist_fp[:,hist_obj]
hist_pre_sig_obj = hist_pre_sig[:,hist_obj]
hist_pre_bkg_obj = hist_pre_bkg[:,hist_obj]



scale_sig = Lumi * hist_sig[:,3] / hist_sig[:,4]
scale_bkg = Lumi * hist_bkg[:,3] / hist_bkg[:,4]
scale_tp = Lumi * hist_tp[:,3] / hist_tp[:,4]
scale_fp = Lumi * hist_fp[:,3] / hist_fp[:,4]
scale_tn = Lumi * hist_tn[:,3] / hist_tn[:,4]
scale_fn = Lumi * hist_fn[:,3] / hist_fn[:,4]
scale_pre_sig = Lumi * hist_pre_sig[:,3] / hist_pre_sig[:,4]
scale_pre_bkg = Lumi * hist_pre_bkg[:,3] / hist_pre_bkg[:,4]


print("##### TP #####")
print(scale_tp.shape,hist_tp_obj.shape)
print("##### FP #####")
print(scale_fp.shape,hist_fp_obj.shape)
print("##### TN #####")
print(scale_tn.shape,hist_tn_obj.shape)
print("##### FN #####")
print(scale_fn.shape,hist_fn_obj.shape)
print("##### SIGNAL #####")
print(scale_sig.shape,hist_sig_obj.shape)
print("##### BKG #####")
print(scale_bkg.shape,hist_bkg_obj.shape)

#### --Prediction distribution
#n_bins=50
#plt.hist(hist_hypo_sig,bins=n_bins,histtype="step",log=True,color='r',label='Signal')
#plt.hist(hist_hypo_bkg,bins=n_bins,histtype="step",log=True,color='b',label='Background')
#plt.legend(loc='upper right')
#plt.title('Prediction')
#axes=plt.gca()
#axes.set_ylim([0,100000])
#plt.savefig('Prediction.png')




#### --Signa and Background
#n_bins=50
#plt.hist(hist_sig_obj,bins=n_bins,weights=scale_sig,histtype="step",log=True,alpha=0.7,color='r',label='Signal')
#plt.hist(hist_bkg_obj,bins=n_bins,weights=scale_bkg,histtype="step",log=True,alpha=0.7,color='b',label='Background')
#plt.legend(loc='upper right')
#plt.title('Mjj')
#plt.savefig('Mjj_FULL_SIG_BKG.png')

#### --Signa and Predicted singal
#n_bins=50
#plt.hist(hist_pre_sig_obj,bins=n_bins,weights=scale_pre_sig,histtype="step",log=True,alpha=0.7,color='r',label='Predicted')
#plt.hist(hist_sig_obj,bins=n_bins,weights=scale_sig,histtype="step",log=True,alpha=0.7,color='b',label='Signal')
#plt.legend(loc='upper right')
#plt.title('Mjj')
#plt.savefig('Mjj_FULL_Predicted.png')

#### --Predicted Signal and Predicted Background
n_bins=50
plt.hist(hist_pre_sig_obj,bins=n_bins,weights=scale_pre_sig,histtype="step",log=True,linewidth=1,alpha=1,color='r',label='DNN_signal')
plt.hist(hist_pre_bkg_obj,bins=n_bins,weights=scale_pre_bkg,histtype="step",log=True,linewidth=1,alpha=1,color='b',label='DNN_bkg')
plt.hist(hist_sig_obj,bins=n_bins,weights=scale_sig,log=True,alpha=0.3,color='r',label='signal')
plt.hist(hist_bkg_obj,bins=n_bins,weights=scale_bkg,log=True,alpha=0.3,color='b',label='bkg')
plt.legend(loc='upper right')
plt.title('Mjj')
plt.savefig('Mjj_FULL_ALL.png')


#### --ROC values on MJJ
#fig,axs = plt.subplots(2,2, figsize=(15,15))
#n_bins=100
#axs[0,0].hist(hist_tp_obj,bins=n_bins,weights=scale_tp ,alpha=0.7,color='r')
#axs[0,0].set_title('Mjj_tp')
#
#axs[0,1].hist(hist_fp_obj,bins=n_bins,weights=scale_fp ,alpha=0.7,color='r')
#axs[0,1].set_title('Mjj_fp')
#
#axs[1,0].hist(hist_tn_obj,bins=n_bins,weights=scale_tn ,alpha=0.7,color='r')
#axs[1,0].set_title('Mjj_tn')
#
#axs[1,1].hist(hist_fn_obj,bins=n_bins,weights=scale_fn ,alpha=0.7,color='r')
#axs[1,1].set_title('Mjj_fn')
#
#plt.savefig('Mjj_FULL.png')

print(" calculated tp: ", len(hist_tp))
print(" calculated tn: ", len(hist_tn))
print(" calculated fn: ", len(hist_fn))
print(" calculated fp: ", len(hist_fp))

print('Accuracy:','{:.2%}'.format(acc_val))
print('AUC:',sess.run(auc, feed_dict={X: test_x_data, Y:test_y_data, keep_prob:1}))
print('TP:', tp)
print('TN:', tn)
print('FP:', fp)
print('FN:', fn)


print("signal: ", test_signal_num)
print("TP+FN : ",tp+fn)
print("BKG   : ", test_bkg_num)
print("FP+TN : ",fp+tn)

print( "Sensitivity: ",tp / (tp+fn))
print( "Specificity: ",tn / (tn+fp))

























