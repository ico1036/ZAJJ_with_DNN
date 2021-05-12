import numpy as np
import pandas as pd
from IPython.display import display

path='../data/data_normal_by_all/'
path_out = '../../Genetic_Algo/data/'

train_data = pd.read_csv(path+'train_data.csv', sep=',')
val_data   = pd.read_csv(path+'val_data.csv', sep=',')
test_data  = pd.read_csv(path+'test_data.csv', sep=',')

train_data.pop('dEtaJJ')
val_data.pop('dEtaJJ')
test_data.pop('dEtaJJ')

train_data.pop('mJJ')
val_data.pop('mJJ')
test_data.pop('mJJ')

train_data['zepp'] = train_data['zepp'].abs()
val_data['zepp']   = val_data['zepp'].abs()
test_data['zepp']  = test_data['zepp'].abs()


# Summary of input features
display(test_data.describe())

np.savetxt(path_out+'train_data.csv',train_data,fmt='%5.5f',header='j1Pt,j2Pt,j1Eta,j2Eta,j1Phi,j2Phi,mJJ,dEtaJJ,zepp,issig',delimiter =',')
np.savetxt(path_out+'val_data.csv',val_data,fmt='%5.5f',header='j1Pt,j2Pt,j1Eta,j2Eta,j1Phi,j2Phi,mJJ,dEtaJJ,zepp,issig',delimiter =',')
np.savetxt(path_out+'test_data.csv',test_data,fmt='%5.5f',header='j1Pt,j2Pt,j1Eta,j2Eta,j1Phi,j2Phi,mJJ,dEtaJJ,zepp,issig',delimiter =',')

