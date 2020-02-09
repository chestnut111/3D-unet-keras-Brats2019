import numpy as np
import os 

name_list = ['t1', 't2', 'flair', 't1ce', 'seg']
fold_train = '/DataProcess/center/allshare/lixin/code/brats2019/MICCAI_BraTS_2019_Data_Training/'
PATCH_SIZE = (128,128,128)
center = "valid" #??
BATCH_SIZE = 4
DATA_SAMPLING = 'all_positive'

# model network 
DEPTH = 5
RESIDUAL = True
DEEP_SUPERVISION = True
FILTER_GROW = True
INSTANCE_NORM = True
NUM_CLASS = 4
BASE_FILTER = 16

#callbacks list
monitor = 'loss'#'val_loss'
mode = 'min'
early_p = 20  #the patience time to stop training while the loss is not down
reduce_lr_p = 10  #the patience time to reduce lr

min_lr = 1e-8
lr = 1e-4



#Keep relevant training record files
time = '2019-10-16-2' 
name = 'one'

path = os.path.join('model_save/', time, name)
if not os.path.isdir(path):
    os.makedirs(path)

log_csv_name = 'log' + '.csv'
log_csv_path = os.path.join(path, log_csv_name)    

best_model_save_name = '{}_weights.best.hdf5'.format('pretrain_model')
best_model_save_path = os.path.join(path, best_model_save_name)

model_save_name = 'model' + '.h5'
model_save_path = os.path.join(path, model_save_name)

conf_save_path = os.path.join(path, 'conf.csv')




