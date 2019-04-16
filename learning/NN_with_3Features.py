import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pandas import Series, DataFrame

## --Read file
train_df=pd.read_csv('../DATA/train_df.csv',sep=" ")
test_df =pd.read_csv('../DATA/test_df.csv',sep=" ")
print("###### train / test Dataframe ######")


## --Make data frame consist of 3 features
train_df_copy = DataFrame(columns=("nttype","ntjjM", "ntjdEta", "ntZpVar"))
test_df_copy  = DataFrame(columns=("nttype","ntjjM", "ntjdEta", "ntZpVar"))

train_df_copy["nttype"]=train_df.nttype
train_df_copy["ntjjM"]=train_df.ntjjM
train_df_copy["ntjdEta"]=train_df.ntjdEta
train_df_copy["ntZpVar"]=train_df.ntZpVar

test_df_copy["nttype"]  =test_df.nttype
test_df_copy["ntjjM"]  =test_df.ntjjM
test_df_copy["ntjdEta"]=test_df.ntjdEta
test_df_copy["ntZpVar"]=test_df.ntZpVar

display(train_df_copy.shape)
display(test_df_copy.shape)


## --Numpy array
print("##### train / test  Numpy array #####")
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
 
train_xy = MinMaxScaler(train_xy)
test_xy  = MinMaxScaler(test_xy)


train_x_data = train_xy[:,1:]
train_y_data = np.rint(train_xy[:,:1])
test_x_data = test_xy[:,1:]
test_y_data = np.rint(test_xy[:,:1])

print("###### Features / Label Numpy ######")
print(train_x_data.shape)
print(train_y_data.shape)
display(train_y_data)

## 

