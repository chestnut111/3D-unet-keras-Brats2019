from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
        
def call(weight_path,  #The best model save path while training , mutil gou model
         monitor,  #loss
         mode,   #min
         reduce_lr_p,  #
         early_p, #
         log_csv_path, # Save log
         save_best_only = True, 
         save_weights_only = True,
         factor=0.5,
         min_delta=0.0001,
         cooldown=2,
         min_lr=1e-6,
         verbose=1
        ):    
        
    csv_logger = CSVLogger(log_csv_path, append=True, separator=',')
    
    checkpoint = ModelCheckpoint(weight_path, 
                                 mode = mode, 
                                 monitor = monitor, 
                                 verbose = verbose, 
                                 save_best_only = save_best_only, 
                                 save_weights_only = save_weights_only)

    reduceLROnPlat = ReduceLROnPlateau(monitor = monitor, 
                                       factor = factor, 
                                       patience = reduce_lr_p, 
                                       verbose = 1, 
                                       mode = mode, 
                                       epsilon = min_delta,  ##
                                       cooldown = cooldown, ##
                                       min_lr = min_lr)

    early = EarlyStopping(monitor = monitor, 
                          mode = mode, 
                          patience = early_p) # probably needs to be more patient
    
    callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]
    return callbacks_list