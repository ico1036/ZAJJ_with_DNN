1. Pre-process step 
  - [Pre-process](https://github.com/ico1036/ZAJJ_with_DNN/tree/master/BK2019_project/preprocess)
  - You can refer any file in that directory, all files include as following:
    - Read ROOT file
    - Divide it train,validation,test sets
    - Shuffle
    - save as 'csv file'
 
2. Training step
  - [Train](https://github.com/ico1036/ZAJJ_with_DNN/tree/master/BK2019_project/training)
  - [Read 'csv inpufile' and do train!](https://github.com/ico1036/ZAJJ_with_DNN/blob/master/BK2019_project/training/train_keras.py)
  - [Monitor the trainig step (loss and accuracy)](https://github.com/ico1036/ZAJJ_with_DNN/blob/master/BK2019_project/training/epoch_loss.py)

3. Evaluation step
  - [Evaluation](https://github.com/ico1036/ZAJJ_with_DNN/tree/master/BK2019_project/evaluation)
  - [DNN score plot and significance](https://github.com/ico1036/ZAJJ_with_DNN/blob/master/BK2019_project/evaluation/Score.py)
  - [ROC curve](https://github.com/ico1036/ZAJJ_with_DNN/blob/master/BK2019_project/evaluation/ROC.py)

---

The root_numpy in "Pre-process" step is deprecated (https://scikit-hep.org/).
So, I recommend you to use the [Uproot](https://github.com/scikit-hep/uproot4) to read the rootfile and processing.
Also, the "csv" file is not efficient. The hdf5 file is more efficient. You can use [h5py](https://docs.h5py.org/en/stable/).  
Here is the example to pre-process the ROOT files to CNN input (.h5 file) using h5py and uproot4.  
https://github.com/nurion4hep/HEP-CNN/blob/master/scripts/makeDetectorImage.py  
  
The basic principle is simple.  
The input-file for training should be looks like this table.  

| label | xsec | Gen-event | features |
|-------|------|-----------|----------|
| 1     |      |           |          |
| 0     |      |           |          |

Since the TensorFlow or Pytorch can read 'csv' or 'hdf5' file, please prepare the file as 'csv' or '.h5'

