# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 18:39:36 2018

@author: n9843329
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import tensorflow.contrib.keras.api.keras as keras
import keras
import math


import random
from keras.models import Sequential
from keras.layers.core import *
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

"""
Required Tasks and Experiments

• Load the MNIST dataset (use keras.datasets.mnist.load_data).
• Split the dataset such that
◦ the digits in [2,3,4,5,6,7] are used for training and testing
◦ the digits in [0,1,8,9] are only used for testing. None of these digits should be used
during training.
• Implement and test the contrastive loss function described earlier in this document.
• Build a Siamese network.
• Train your Siamese network on your training set. Plot the training and validation error vs
time.
• Evaluate the generalization capability of your network by
◦ testing it with pairs from [2,3,4,5,6,7] x [2,3,4,5,6,7]
◦ testing it with pairs from [2,3,4,5,6,7] x [0,1,8,9]
◦ testing it with pairs from [0,1,8,9] x [0,1,8,9]
• Present your results in tables and figures


"""

###############################################################################
#                      contrastive_loss
###############################################################################


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss 
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


###############################################################################
#              Skeleton of the Neural Network Architecture
# -----------------------------------------------------------------------------
# 
# Parameters to be determined    
###############################################################################

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_pairs_train(x, digit_indices, train_digits):
    """
    Create training pairs (positive and negative) and corresponding y value of the training pairs
    
    Param:
                x: The raw training samples 
    digit_indices: The matrix which records the indices of all the digits to be kept
     train_digits: The digits to be kept in the training set
     
     Return
     np.array(pairs): An array of image pairs 
    np.array(labels): An array of corresponding labels indicating whether it's a positive pair or negative pair
    
    """
    
    # Create empty list of pairs and labels to be appended
    pairs = []
    labels = []
    
    # calculate the min number of training sample of each digit in training set
    min_sample = [len(digit_indices[d]) for d in range(len(train_digits))]
    
    # calculate the number of pairs to be created
    n = min(min_sample) -1
    
    # Looping over each digits in the train_digits
    for d in range(len(train_digits)):
        
        # Create n pairs of same digits and then create the same amount of pairs for the different digits
        for i in range(n):
            
            # Create a pair of same digits: 
            # retrieve the index of a pair of same digit 
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            # Append the image pair of same digits to the pair list
            pairs += [[x[z1], x[z2]]]
            
            # Create a pair of different digits
            # First create a randome integer rand falls between (1, len(train_digits))
            # let dn be (d+rand) % len(train_digit) so that dn will distinct from d 
            # and that is guaranteed to be a different digits
            rand = random.randrange(1, len(train_digits))
            dn = (d + rand) % len(train_digits)
            
            # Use the dn and d to create a pair of different digits 
            # the append it to the pair list
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            # Append the corresponding label value for the true and false pairs of image created
            labels += [1, 0]
    
    return np.array(pairs), np.array(labels)




    
def create_cnn(input_shape):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=input_shape))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2)) #0.125
    
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2)) #0.25
    model.add(keras.layers.Flatten())
 
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(2, activation='sigmoid'))
    
    
    return model
    

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = keras.layers.Input(shape=input_shape)
    
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    return keras.models.Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def display_image(image):
    plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.show()


###############################################################################
#                          Data Preparation 
###############################################################################

# Call the Keras.datasets.mnist API and split the data into training and 
# test set where training sets contain 60000 (28*28) images and test set 
# contains 10000 (28*28) images
digits_to_keep = [2,3,4,5,6,7]
digits_to_be_removed = [0, 1, 8, 9]
all_digits = digits_to_keep + digits_to_be_removed

def get_data():
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Ensure the data type is float instead of int
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Normalize the matrix
    X_train /= 255
    X_test /= 255
    
    # Reshape the image to desinated input shape for NN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1 )
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1 )
    
    # Each input data should be in the form of (28,28,1)
    input_shape = X_train.shape[1:]
    
    # Concatenate the X and y data, then split the data into 80-20 proportion
    # as per requirement
    X_all = np.append(X_train, X_test, axis = 0)
    y_all = np.append(y_train, y_test, axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, 
                                                        random_state=42)
    
    
    # Extract [0, 1, 8, 9] from training set and concatenate them to the test set
    digits_to_keep = [2,3,4,5,6,7]
    digits_to_be_removed = [0, 1, 8, 9]
    all_digits = digits_to_keep + digits_to_be_removed
    
    # Create a mask index to extract the digits to be kept
    mask = np.array([True if i in digits_to_keep else False for i in y_train])
    
    # keep the digits to be kept in training set only
    revised_X_train = X_train[mask]
    revised_y_train = y_train[mask]
    
    # Append the removed data to the testing set
    revised_X_test = np.append(X_test, X_train[~mask], axis = 0)
    revised_y_test = np.append(y_test, y_train[~mask], axis = 0)
    
    return input_shape, revised_X_train, revised_y_train, revised_X_test, revised_y_test

input_shape, revised_X_train, revised_y_train, revised_X_test, revised_y_test = get_data()

###############################################################################
#                        Training Pairs Creation
###############################################################################

# Now the training set contains only the digits in digits_to_keep[2,3,4,5,6,7]
# To train a siamese NN, we need to create samples of true pairs and false pairs
# where the True pairs are pairs of same digits (2,2), (3,3), (4,4), ..., (7, 7)
# whereas the False pairs are pairs of different digits (2, 3), (4, 6), (6, 7)..

# First, create a digit-index matrix where each row is the indices of a particular digits in the training set
# 
# digit 2 [ ind, ind, ind,...............ind]
# digit 3 [ ind, ind, ind,...............ind] 
# digit 4 [ ind, ind, ind,...............ind]
# digit 5 [ ind, ind, ind,...............ind]
# digit 6 [ ind, ind, ind,...............ind]
# digit 7 [ ind, ind, ind,...............ind]
digit_indices = [np.where(revised_y_train == j)[0] for i, j in enumerate(digits_to_keep)]

# Create training pairs (tr_pairs) and the corresponding y value (tr_y)
# Note the training set contains only the digits_to_keep [2,3,4,5,6,7]
tr_pairs, tr_y = create_pairs_train(revised_X_train, digit_indices, digits_to_keep)
"""
for i in range(5):
    display_image(tr_pairs[2000*i+1][0])
    plt.show()
    display_image(tr_pairs[2000*i+1][1])
    plt.show()    
    
    print(tr_y[i*2000+1])
 """
###############################################################################
#                Create testing set for experiment 1
# -----------------------------------------------------------------------------
#              Experiment 1 [2,3,4,5,6,7] vs [2,3,4,5,6,7]
###############################################################################
# Construct a subset from test set that only contains the digits of [2,3,4,5,6,7]
# then create pairs of testing data and the corresponding labels
mask = [True if i in digits_to_keep else False for i in revised_y_test]
exp_1_X_test = revised_X_test[mask]
exp_1_y_test = revised_y_test[mask]

# create the digit-index matrix where each row is the indices of a particular digits in the training set
digit_indices = [np.where(exp_1_y_test == j)[0] for i, j in enumerate(digits_to_keep)]
"""
Debuging purpose - plot out the image for debug purpose

for j in range(len(digits_to_keep)): 
    
    for i in range(10):
        display_image(exp_1_X_test[digit_indices[j][-i]])
        plt.show()

"""

# Create testing pairs and the corresponding y value for experiment 1
exp_1_pairs, exp_1_y = create_pairs_train(exp_1_X_test, digit_indices, digits_to_keep)
"""
Debuging purpose - plot out the image for debug purpose

for i in range(5):
    display_image(exp_1_pairs[3005*i][0])
    plt.show()
    display_image(exp_1_pairs[3005*i][1])
    plt.show()    
    
    print(exp_1_y[i*3005])
"""
###############################################################################
#                Create testing set for experiment 2
# -----------------------------------------------------------------------------
#              Experiment 2 [0,1,8,9] vs [0,1,8,9]
###############################################################################

# Construct a subset from test set that only contains the digits of [0,1,8,9]
# then create pairs of testing data and the corresponding labels
mask = [True if i in digits_to_be_removed else False for i in revised_y_test]
exp_2_X_test = revised_X_test[mask]
exp_2_y_test = revised_y_test[mask]

# create the digit-index matrix where each row is the indices of a particular digits in the training set
digit_indices = [np.where(exp_2_y_test == j)[0] for i, j in enumerate(digits_to_be_removed)]

# Create testing pairs and the corresponding y value for experiment 2
exp_2_pairs, exp_2_y = create_pairs_train(exp_2_X_test, digit_indices, digits_to_be_removed)


###############################################################################
#                Create testing set for experiment 3
# -----------------------------------------------------------------------------
#              Experiment 3 [2,3,4,5,6,7] union [0,1,8,9]
###############################################################################

# create the digit-index matrix where each row is the indices of a particular digits in the training set
digit_indices = [np.where(revised_y_test == j)[0] for i, j in enumerate(all_digits)]


# Create testing pairs and the corresponding y value for experiment 3
exp_3_pairs, exp_3_y = create_pairs_train(revised_X_test, digit_indices, all_digits)


"""
Debuging purpose - plot out the image for debug purpose

for i in range(10):
    display_image(exp_3_pairs[801*i][0])
    plt.show()
    display_image(exp_3_pairs[801*i][1])
    plt.show()    
    
    print(exp_3_y[i*801])
"""

###############################################################################
batch_size = 128
epochs = 10

def get_model(tr_pairs):

    # Define input shape
    img_rows, img_cols = tr_pairs.shape[2:4]
    
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    tr_pairs = tr_pairs.reshape(tr_pairs.shape[0], tr_pairs.shape[1], img_rows, img_cols, 1)
    
    
    
    #input_shape = (img_rows, img_cols, 1)
    
    # define batch size and epochs

    
    # Defining Siamese neural network
    base_network = create_cnn(input_shape)
    
    input_a = keras.layers.Input(shape=input_shape)
    input_b = keras.layers.Input(shape=input_shape)
    
    
    # reuse the instance of the neural network so that the weights of the neuralnet 
    # will be shared accross the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = keras.layers.Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model_1 = keras.models.Model([input_a, input_b], distance)
    
    # Different optimizers
    rms = keras.optimizers.RMSprop()
    adam = keras.optimizers.Adam()

    
    # Model training
    model_1.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    return model_1

model = get_model(tr_pairs)

###############################################################################
#                   Model fitting and evaluation
#------------------------------------------------------------------------------
#
###############################################################################


def fit_model(model, tr_pairs, tr_y, test_pairs_1, test_y_1, test_pairs_2, test_y_2, test_pairs_3, test_y_3):
    """
    Fit the model with training data, evaluate the model then execute the experiment 1-3
    
    Param:
        model: The CNN Siamese network          
        tr_pairs: The training pairs
        tr_y: The label of the training data
        test_pairs_1: The testing pairs for experiment 1 - [2,3,4,5,6,7] vs [2,3,4,5,6,7]
        test_y_1: The labels of the testing data of experiment 1
        test_pairs_2: The testing pairs for experiment 2 - [0,1,8,9] vs [0,1,8,9]
        test_y_2: The labels of the testing data of experiment 2
        test_pairs_3: The testing pairs for experiment 3 - [2,3,4,5,6,7] union [2,3,4,5,6,7]
        test_y_3: The labels of the testing data of experiment 3
    
    Output:
        Training accuracy and the test accuracy of the 3 corresponding experiments    
    
    """
    
    # Reserve 20% data for validatoin
    X_trn, X_val, y_trn, y_val = train_test_split(tr_pairs, tr_y,
                                                  stratify = tr_y, 
                                                  test_size = 0.2)
    
    # Fit the model with training data and validate on the reserved 20% validation data
    model.fit([X_trn[:, 0], X_trn[:, 1]], y_trn,
              batch_size=128,
              epochs=epochs,
              validation_data=([X_val[:, 0], X_val[:, 1]], y_val))
    
  
    # compute final accuracy on training and test sets
    y_pred = model.predict([X_trn[:, 0], X_trn[:, 1]])
    tr_acc = compute_accuracy(y_trn, y_pred)
    y_pred = model.predict([test_pairs_1[:, 0], test_pairs_1[:, 1]])
    te_acc_1 = compute_accuracy(test_y_1, y_pred)
    
    y_pred = model.predict([test_pairs_2[:, 0], test_pairs_2[:, 1]])
    te_acc_2 = compute_accuracy(test_y_2, y_pred)
    
    y_pred = model.predict([test_pairs_3[:, 0], test_pairs_3[:, 1]])
    te_acc_3 = compute_accuracy(test_y_3, y_pred)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set_1: %0.2f%%' % (100 * te_acc_1))
    print('* Accuracy on test set_2: %0.2f%%' % (100 * te_acc_2))    
    print('* Accuracy on test set_3: %0.2f%%' % (100 * te_acc_3))



def sampling(X, y, shrink_portion, max_constraint = False, constraint_size = 1500):
    """
    Sample the given data (X and y) according to the given portion to create subset 
    of the given data that require less time for training and testing
    
    Param:
        X: Image pairs
        y: lables 
        shrink_portion: A fraction that decides the size of the sampled data
        max_constraint: (Bool) Limit the number of samples if True
        Constraint_size: maximum sample size if mex_constraint is True 
    
    Return: 
        Sampled, shuffled image pairs and labels

    """    
    
    
    # The index of positive pairs in the given image pairs X
    mask_positive = [i for i in range(X.shape[0]) if i % 2 ==0]
    # The index of negative pairs in the given image pairs X 
    mask_negative = [i for i in range(X.shape[0]) if i % 2 !=0]
    
    # Calculate the totla number of observations according to the given shrink_portion
    sample_num = int(math.floor(X.shape[0]*shrink_portion))

    # If maximum size constraint applies
    if max_constraint:
        # Set the sample size to be constraint_size (default 1500)
        sample_num = constraint_size
        
    # To ensure the positive and negative pairs are equally appeared in the training set
    # sample size -= 1 if it's an odd number
    if sample_num % 2!=0:
        sample_num -= 1
    
    # The index of randomly selected (sample size/2) positive pairs 
    positive_mask = np.random.choice(len(mask_positive), sample_num // 2, shrink_portion)
    # The index of randomly selected (sample size/2) negative pairs 
    negative_mask = np.random.choice(len(mask_negative), sample_num // 2, shrink_portion)
    
    # Concatenate the index of the sampled data (positive pairs and negative pairs)
    index = np.append(positive_mask, negative_mask)
    index = np.sort(index)
    
    # Return the sample pairs of images and labels
    return X[index], y[index]




# Sample data for training/testing
    
# Sample 30% of the training data (image pairs and labels) for training purpose 
sampled_train_pairs, sampled_train_y = sampling(tr_pairs, tr_y, 0.3)

# Sample 30% testing data (images and labels) for experiment 1 and limit the test size to be 1500 at most
sampled_exp_1_pairs, sampled_exp_1_y = sampling(exp_1_pairs, exp_1_y, 0.3, True)

# Sample 30% testing data (images and labels) for experiment 2 and limit the test size to be 1500 at most
sampled_exp_2_pairs, sampled_exp_2_y = sampling(exp_2_pairs, exp_2_y, 0.3, True)

# Sample 30% testing data (images and labels) for experiment 3 and limit the test size to be 1500 at most
sampled_exp_3_pairs, sampled_exp_3_y = sampling(exp_3_pairs, exp_3_y, 0.3, True)

# Shuffle the index of the training data and test data (same index applies to image pairs and labels)
sampled_train_pairs, sampled_train_y = shuffle(sampled_train_pairs, sampled_train_y)
sampled_exp_1_pairs, sampled_exp_1_y = shuffle(sampled_exp_1_pairs, sampled_exp_1_y)
sampled_exp_2_pairs, sampled_exp_2_y = shuffle(sampled_exp_2_pairs, sampled_exp_2_y)
sampled_exp_3_pairs, sampled_exp_3_y = shuffle(sampled_exp_3_pairs, sampled_exp_3_y)


###############################################################################
#                                Experiments
#------------------------------------------------------------------------------
#
#
###############################################################################
# Fit the model, validate on validation set and evaluate the test data for 3 experiments


fit_model(model, sampled_train_pairs, sampled_train_y, 
          sampled_exp_1_pairs, sampled_exp_1_y, 
          sampled_exp_2_pairs, sampled_exp_2_y, 
          sampled_exp_3_pairs, sampled_exp_3_y)




def plot(model, X_tr, y_tr, X_val, y_val): 
    """
    Train, Validate and Plot the training, validatoin accuracy, loss over epoch
    
    Param:
        model: NN Model
        X_tr: X_train
        y_tr: y_train
        X_val: X_validation
        y_val: y_validation
    
    Output:
        train/val Accuracy over epoch
        train/val Loss over epoch
    """
    history = model.fit([X_tr[:,0],
                         X_tr[:,1]], 
                         y_tr, 
                         epochs=10, batch_size=128, verbose=1, 
                         validation_data=([X_val[:, 0], X_val[:, 1]], y_val))

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    



def k_fold_cross_validation(k, model, X, y): 
    """
    k-fold cross validation and write the outcome to a txt file
    
    Param
        k: number of folds 
        model: the model
        X: X (image pairs)
        y: label
        
    Output:
        cross_validation result 
    """
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state = 42)
    cv_scores = []
    for train, test in kfold.split(X, y):
  
    	model.fit([X[train, 0], X[train, 1]], y[train], epochs=10, batch_size=128, verbose=1)
    	# evaluate the model
    	scores = model.evaluate([X[test, 0], X[test, 1]], y[test], verbose=1)
    	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    	cv_scores.append(scores[1] * 100)
    print("%.2f%% (std %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    np.savetxt('test.txt', cv_scores, delimiter=',')




k_fold_cross_validation(10, model, sampled_train_pairs, sampled_train_y)

    

"""
16/32/64/0.25/10/rms

* Accuracy on training set: 89.35%
* Accuracy on test set: 84.36%

* Accuracy on training set: 90.87%
* Accuracy on test set: 65.14%

* Accuracy on training set: 93.50%
* Accuracy on test set: 75.14%


32/64/128/0.25/10/rms/with softmax
* Accuracy on training set: 91.17%
* Accuracy on test set: 86.29%

* Accuracy on training set: 95.83%
* Accuracy on test set: 63.01%

* Accuracy on training set: 98.18%
* Accuracy on test set: 73.93%



32/64//128/0.25/10/rms/relu


* Accuracy on training set: 99.83%
* Accuracy on test set: 99.33%

* Accuracy on training set: 99.89%
* Accuracy on test set: 61.27%

* Accuracy on training set: 99.94%
* Accuracy on test set: 79.00%

"""


"""
32/64/128/0.25/10/adam/all relu
* Accuracy on training set: 99.93%
* Accuracy on test set: 98.73%

* Accuracy on training set: 99.94%
* Accuracy on test set: 61.80%

* Accuracy on training set: 99.95%
* Accuracy on test set: 79.07%
"""






