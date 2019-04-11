import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from ROOT import *
from root_numpy import root2array, tree2array
from root_numpy import testdata
from IPython.display import display

## --Conver Ntuple to Pandas Dataframe
infile 		= TFile.Open('ALL.root')
file_tree   = infile.Get('tree')
data_arr 	= tree2array(file_tree)
data_df  	= pd.DataFrame(data_arr)


## --Make signal and background dataframe
signal_df=data_df[data_df['nttype'] == 1]
bkg_sample=data_df[data_df['nttype'] == 0]


## --Random sampling for signal
from sklearn.model_selection import train_test_split
signal_rest, signal_sample = train_test_split(signal_df, test_size=0.1214199, random_state=42)
display(signal_sample.info())

## --Signal Mjj plots(origin and sample) for validation
#fig,axs = plt.subplots(1,2, figsize=(10,6))
#n_bins=20
#
#axs[0].hist(signal_df['ntjjM'],bins=n_bins,alpha=0.7,color='r')
#axs[0].set_title('Mjj_origin')
#
#axs[1].hist(signal_sample['ntjjM'],bins=n_bins,alpha=0.7,color='r',)
#axs[1].set_title('Mjj_sample')
#plt.savefig('Mjj_signal.png')




## --Combine data for analysis
display(signal_sample.shape)
display(bkg_sample.shape)

Anal_df=pd.concat([signal_sample,bkg_sample])
del Anal_df["ntgenN"]
del Anal_df["ntxsec"]
display(Anal_df.info())
Anal_df.to_csv('Anal.csv', index=False, sep=' ')


## --Hierarchical sampling
#from sklearn.model_selection import StratifiedShuffleSplit
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
#for rest_index, sample_index in split.split(signal_df, signal_df["ntjjM"]):
#    signal_sample_set = signal_df.loc[sample_index]
#    signal_rest_set   = signal_df.loc[rest_index]



