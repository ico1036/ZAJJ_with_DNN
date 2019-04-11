import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# --Read file
data_df=pd.read_csv('Anal.csv',sep=" ")
#display(Anal_df)

## --Make signal and background dataframe
signal_df=data_df[data_df['nttype'] == 1]
bkg_df=data_df[data_df['nttype'] == 0]

display(signal_df.shape)
display(bkg_df.shape)

## --Random sampling for signal
from sklearn.model_selection import train_test_split

signal_train_df, signal_test_df = train_test_split(signal_df, test_size=0.3, random_state=42)
bkg_train_df, bkg_test_df = train_test_split(bkg_df, test_size=0.3, random_state=42)

print("Signal #####")
display(signal_train_df.shape)
print("BKG    #####")
display(bkg_train_df.shape)

# --Signal Mjj plots(origin and sample) for validation
#fig,axs = plt.subplots(2,2, figsize=(15,15))
#n_bins=100
#
#axs[0,1].hist(signal_train_df['ntjjM'],bins=n_bins,alpha=0.7,color='r')
#axs[0,1].set_title('Mjj_signal_train')
#
#axs[0,0].hist(signal_df['ntjjM'],bins=n_bins,alpha=0.7,color='r')
#axs[0,0].set_title('Mjj_signal_origin')
#
#axs[1,1].hist(bkg_train_df['ntjjM'],bins=n_bins,alpha=0.7,color='b',)
#axs[1,1].set_title('Mjj_bkg_train')
#
#axs[1,0].hist(bkg_df['ntjjM'],bins=n_bins,alpha=0.7,color='b',)
#axs[1,0].set_title('Mjj_bkg_origin')
#
#plt.savefig('Mjj_train.png')


## --Make training dataset
print("Training set")
train_df=pd.concat([signal_train_df,bkg_train_df])
display(train_df.shape)

from sklearn.utils import shuffle
train_df=shuffle(train_df)
#display(train_df)


## --Correlation plot
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#x=train_df.drop('nttype',axis=1)
#f,ax = plt.subplots(figsize=(18, 18))
#sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#plt.savefig('corr.png')

print("Training set after feature reduction")
train_df_copy = train_df.drop('ntele2Eta',1)
train_df_copy = train_df_copy.drop('ntjet2PT',1)
train_df_copy = train_df_copy.drop('ntdRjj',1)
display(train_df_copy.shape)

