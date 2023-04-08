import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

import pandas as pd
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pickle
import os


#######################################################################
#---------------------------------------------------------------------#
#---------------   DEEP LEARNING FUNCTIONS   -------------------------#
#---------------------------------------------------------------------#
#######################################################################

#--------------------------------------------------------------------------------   
def display_model(model, path_save_fig):

    print(f'==> Saving model display at {path_save_fig}')

    fig = plt.figure(figsize=(12,8))
    tf.keras.utils.plot_model(model, path_save_fig + '.png', show_shapes=True)
    #plt.show()
    
#--------------------------------------------------------------------------------       
def load_one_image_to_make_prediction(img_path):
    
    img = mpimg.imread(img_path)
    img = img[...,np.newaxis]
    
    x = image.img_to_array(img)
    #print(x.shape) #(256, 256, 1)
    x = np.expand_dims(x, axis=0)
    #print(x.shape) #(1, 256, 256, 1)
    
    return(x)
#--------------------------------------------------------------------------------   

def get_model(input_shape):
    
    model_name = 'model'
    
    l_init = Input(shape=input_shape, name='input')
    
    x = Conv2D(filters=64, kernel_size=(8,8), activation='relu', padding='same', name='conv_1')(l_init)
    x = MaxPooling2D(pool_size=(2,2), name='max_pooling_1')(x)
    x = Dropout(0.3,name='dropout_1')(x)
    
    x = Conv2D(filters=64, kernel_size=(4,4), activation='relu', padding='same', name='conv_2')(x)
    x = MaxPooling2D(pool_size=(2,2), name='max_pooling_2')(x)
    x = Dropout(0.3,name='dropout_2')(x)
    
    x = Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', name='conv_3')(x)
    x = MaxPooling2D(pool_size=(2,2), name='max_pooling_3')(x)
    
    x = Flatten(name='flatten')(x)
    
    x = Dense(32, activation='relu', name='dense_1')(x)
    x = Dropout(0.3,name='dropout_3')(x)

    l_final = Dense(1,activation='sigmoid', name='output')(x)
    
    model = Model(inputs=[l_init],outputs=[l_final])
    
    model.compile(optimizer = Adam(learning_rate=0.0005), 
                  loss = tf.keras.losses.BinaryCrossentropy(from_logits=False), 
                  metrics=['acc',tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision')])
    return(model,model_name)

#--------------------------------------------------------------------------------
def train_model(model, train_gen, val_gen, epochs, checkpoint_path, early_stop_patience, reduce_lr_patience):
    '''
    Trains the model for the number of epochs especified above, using the callbacks EarlyStopping, 
    ReduceLROnPlateau and ModelCheckpoint
    
    - At the end of every epoch, the model weights are saved into a directory called checkpoint_path 
        only if those weights lead to the highest validation recall
    - Training stops when the validation metric has not improved in patience epochs.
    '''
    
    print('==> Train Model - Begin')
    #callback that stops training when the validation metric has not improved in the last x epochs.
    early_stop = EarlyStopping(patience=early_stop_patience,monitor='val_loss', min_delta=0.001,mode='min',verbose=1) 
    #Reduce learning rate when a metric has stopped improving.
    reduce_lr = ReduceLROnPlateau(patience=reduce_lr_patience ,monitor='val_loss',min_delta=0.001, mode='min',factor=0.5, min_lr=1e-4,verbose=1) 
    # Save best model
    checkpoint = ModelCheckpoint(filepath=checkpoint_path + '_best_model.h5', save_weights_only=False, save_best_only=True, monitor='val_recall', mode='max',
                                 save_frequency='epoch',verbose=1)

    cb = [early_stop,reduce_lr,checkpoint]
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cb)

    print('==> Train Model - End')
    print(f'==> Best model saved at {checkpoint_path}')
    return(history)

#--------------------------------------------------------------------------------
def get_saved_model(model_path):
    '''
    Returns a saved model at path model_path
    '''
    model = load_model(model_path)
    
    return(model)

#--------------------------------------------------------------------------------
def get_test_accuracy(model, x_test, y_test):
    '''
    Test model classification accuracy
    '''
    
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('==> accuracy: {acc:0.3f}'.format(acc=test_acc))
    
    return(test_loss,test_acc)
    
#--------------------------------------------------------------------------------
def get_ImageDataGenerator():
    '''
    Create a data generator using the ImageDataGenerator class.
    * Scales the image pixel values by a factor of 1/255
    '''
    image_generator = ImageDataGenerator(
                        rescale=(1/255.))
    
    return(image_generator)

#--------------------------------------------------------------------------------
def get_ImageDataGenerator_augmented():
    '''
    Create a data generator using the ImageDataGenerator class
    * Scales the image pixel values by a factor of 1/255
    * Randomly rotates images by up to 30 degrees
    * Randomly alters the brightness (picks a brightness shift value) from the range (0.5, 1.5)
    * Randomly flips images horizontally
    '''
    image_generator = ImageDataGenerator(
                        rescale=(1/255.),
                        rotation_range = 5,
                        horizontal_flip = True,
                        brightness_range = (0.5,1.5))
    
    return(image_generator)

#--------------------------------------------------------------------------------

def get_generator(image_generator, directory, classes, t_size, b_size=32, shuffle=True, seed=None):
    '''
    Returns a generator object that returns batches of images and labels from the directories.
    color_mode = 'gray_scale': returns only one color channel
    class_mode = 'sparse': returns 1d integer labels. "categorical": returns 2d one-hot encoded labels
    classes: ['Not-COVID','COVID-19'], sparse will return [0,1] (0='Not-COVID', 1='COVID-19')
    '''
    
    image_generator_iterable = image_generator.flow_from_directory(
                                                directory, batch_size = b_size, classes = classes, target_size = (t_size,t_size),
                                                seed=seed, class_mode='sparse', shuffle = shuffle, color_mode = 'grayscale'
                                                )
    #filenames = image_generator_iterable.filenames
    return(image_generator_iterable)

#--------------------------------------------------------------------------------

def save_history(history,path):
    
    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("==> hitory.history saved in ", path)

#--------------------------------------------------------------------------------

def load_history(path):
    
    with open(path, "rb") as file_pi:
        history = pickle.load(file_pi)
    
    return(history)

#--------------------------------------------------------------------------------

def sigmoid_prediction_to_binary_class(predictions):
    y_pred = predictions.copy().flatten()
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    y_pred = y_pred.astype(int)
    
    return(y_pred)

#--------------------------------------------------------------------------------

#######################################################################
#---------------------------------------------------------------------#
#---------------   EXPLORATORY DATA ANALISYS   -----------------------#
#---------------------------------------------------------------------#
#######################################################################

def create_df_with_images_paths_and_targets(main_path,folders):

    df = []
    for i, level in enumerate(folders):
        data_path = main_path + level + '/images'
        for file in os.listdir(data_path):
            img_path = data_path + '/'+ file
            img = mpimg.imread(img_path)
            df.append([img_path, img, level,i])

    df = pd.DataFrame(df,columns=['file','image','result','result_index'])    

    return(df)


    