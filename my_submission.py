'''
---------------------------------- START OF DOCUMENT ----------------------------------
'''

'''

my_submission.py file that is the developed code for Assignment #2 in Unit IFN680 at Queensland University of Technology in Brisbane during Semester 2, 2018.

Due date for submission is the 28th of October @ 11.59pm

This assignment is submitted by:
    Max Edman       n10156003
    Alex Kamrath    
    David Ding Lu   

'''

# Imports modules to complete the Assignment
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import random

# Imports required modules from the Keras Functional API
import keras
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model

'''
Defining static final variables to be used throughout the assignment.
'''
# Defining a set of static variables that are used throughout the defined functions.
batch_size = 128

# How many epochs that should be run for the siamese model
epochs_siamese = 30

# In the MNIST set the number of classes/labels/digits is 10.
num_classes = 10

# Defining ranges with for different test sets based on the target label of the different pictures.
set1_digits = [2,3,4,5,6,7]
set2_digits = [0,1,8,9]
set3_digits = [0,1,2,3,4,5,6,7,8,9]


'''
------------------------------------------------------------------------------
List of functions in this .py document:
+ save_mnist_dataset()
+ load_mnist_dataset()
+ preprocess_mnist_dataset(x_train, y_train, x_test, y_test)
+ reshape_convert_input_data(input_data)
+ build_CNN(input_shape)
+ euclidean_distance(vects)
+ eucl_dist_output_shape(shapes)
+ contrastive_loss_function(y_true, y_pred)
+ create_pairs_set(x, digit_indices, test_index)
+ compute_accuracy(y_true, y_pred)
+ accuracy(y_true, y_pred)
+ siamese_network()

------------------- END OF LIST OF FUNCTIONS ---------------------------------
'''


# Defining function to save the dataset to be used in Assignment #2 in Unit IFN680 @ QUT
def save_mnist_dataset():
    '''
    Loads the MNIST dataset and saves it locally.
    '''
    from tensorflow.contrib import keras
    from keras.datasets import mnist
    
    # Loads the MNIST-dataset from the keras library.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Saves the dataset locally as a .npz file.
    np.savez('mnist_dataset.npz',
             x_train = x_train,
             y_train = y_train,
             x_test = x_test,
             y_test = y_test)

# Function to load the mnist dataset    
def load_mnist_dataset():
    '''
    Loads the local dataset
        -returns: the dtaset
    '''
    # Loads the dataset from the locally saved .npz file.
    with np.load('mnist_dataset.npz') as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']
    
    return x_train, y_train, x_test, y_test


def preprocess_mnist_dataset(x_train, y_train, x_test, y_test):
    '''
    # Function to implement the preprocessing steps of:
    #   Split the dataset such that ...
    #   ◦ the digits in [2,3,4,5,6,7] are used for training and testing
    #   ◦ the digits in [0,1,8,9] are only used for testing. None of these digits should be used during training.
    Returns a total of 4 different datasets: one for training and 3 for testing. As specified in the assignment outline.
    
    -args:
        x_train     Input training data
        y_train     Target for training data
        x_test      Input test data
        y_test      Target for test data
        
    -returns:
        (data_train, target_train)              Dataset for training the model
        (data_test, target_test)                Dataset 1 for testing
        (only_test_dataset, only_test_target)   Dataset 2 for testing
        (final_test_data, final_test_target)    Dataset 3 for testing
    '''
    
    # Using numpy module to concatenate train and test data into their own datasets. Respectively for x and y
    image_dataset = np.concatenate([x_train, x_test])
    target_dataset = np.concatenate([y_train, y_test])
    
    # Creates a mask with all numbers that should be used for both training and testing. I.e. where target dataset is any of the following numbers [2,3,4,5,6,7]. Will be further called tnt_dataset
    train_and_test_mask = np.logical_and(target_dataset>1, target_dataset<8)
    
    # Initiating two new arrays with data and target based on previous created mask. These will be used for both testing and traning.
    #tnt_dataset = np.ma.array(image_dataset, mask=train_and_test_mask)
    test_and_train_dataset = image_dataset[train_and_test_mask,:,:]
    test_and_train_target = target_dataset[train_and_test_mask]
    
    # Initiating two new arrays with data and target based on the remaining entries in the array. I.e. the negative of the mask. These will ONLY be used for testing.
    only_test_dataset = image_dataset[~train_and_test_mask,:,:]
    only_test_target = target_dataset[~train_and_test_mask]
    
    # Import module for splitting datasets.
    from sklearn.model_selection import train_test_split
    # Splits the dataset that are used for both train and test into respective sets. With test size of 20% as stated in the implementation hints "Keep 80% of the [2,3,4,5,6,7] digits for training (and 20% for testing)."
    data_train, data_test, target_train, target_test = train_test_split(test_and_train_dataset,
                                                                        test_and_train_target,
                                                                        test_size=0.20)
    
    # Concatenates the data that should be used for testing.
    final_test_data = np.concatenate([data_test, only_test_dataset])
    final_test_target = np.concatenate([target_test, only_test_target])
    
    # Returns the data to be used in training (80% of the digits [2,3,4,5,6,7]) and testing (the rest).
    return (data_train, target_train), (data_test, target_test), (only_test_dataset, only_test_target) , (final_test_data, final_test_target)

    
def reshape_convert_input_data(input_data):
    '''
    Reshapes and convert data from argument input_data. The input_data is reshaped to a 4D array to be used as input for the model. The input data is converted by normalisation, from RGB 0-255 to 0-1.
    
    -args:
        input_data      Data to be converted
    
    -returns:
        input_data      The converted dataset
    '''
    # Gets the dimensions of the input images
    img_rows, img_cols = input_data.shape[1:3]
    
    # reshape the input arrays to 4D (amount of images, rows, columns, channels)
    input_data = input_data.reshape(input_data.shape[0], img_rows, img_cols, 1)
    
    # convert to float32 and rescale between 0 and 1 to normalise the image data.
    # All the data is an RGB number between 0 and 255. Which is why dividing the vectors with 255 rescales the input data between 0 and 1.
    input_data = input_data.astype('float32')
    input_data /= 255
    
    # Returns the converted and reshaped data
    return input_data
    
    
def build_CNN(input_shape):
    '''
    Build a CNN model to be used as a shared network in the siamese network model.
    Mainly copied from the CNN practical during week 7.
    
    -args:
        input_shape     The dimenstions of the dataset to be used
        
    -returns:
        cnn_model       A keras Sequential model
    '''
    
    # Initiates a sequential model
    cnn_model = keras.models.Sequential()
    
    # Adds layers to the sequential model
    cnn_model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    cnn_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(keras.layers.Dropout(0.25))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(128, activation='relu'))
    cnn_model.add(keras.layers.Dropout(0.5))
    cnn_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    # Retunrs the specified sequential model, based on the assignment outline.
    return cnn_model


def euclidean_distance(vects):
    '''
    Alex
    
    -args:
        vects       *DESCRIPTION*
    
    -returns:
        *DESCRIPTION*
    '''
    
    # ALEX
    x, y = vects
    
    # ALEX
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    
    # ALEX
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    '''
    ALEX
    
    -args:
        shapes       *DESCRIPTION*
    
    -returns:
        *DESCRIPTION*
    '''
    
    # ALEX
    shape1, shape2 = shapes
    
    # ALEX
    return (shape1[0], 1)


def contrastive_loss_function(y_true, y_pred):
    '''
    Contrastive loss 
    ALEX
    
    -args:
        y_true       *DESCRIPTION*
        y_pred       *DESCRIPTION*
    
    -returns:
        *DESCRIPTION*
    '''
    
    # The margin m > 0 determines how far the embeddings of a negative pair should be pushed apart.
    m = 2 # margin # Might need to be changed and evaluated for what value the siamese network performs best.

    # ALEX
    sqaure_pred = K.square(y_pred)
    
    # ALEX
    margin_square = K.square(K.maximum(m - y_pred, 0))
    
    # ALEX
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs_set(x, digit_indices, test_index):
    '''
    Positive and negative pair creation.
    
    Creates an array of positive and negative pairs combined with their label (1 or 0) - depending on if the two images used as input is considered to be from the same eqivalence class then they are considered a positive pair. If they are not, they are considered a negative pair.
    
    -args:
        x               Dataset from where pairs are to be created.
        digit_indices   ALEX
        test_index      Index of 1 to 3 depending on what dataset has been provided as x
    
    -returns:
        An array containing the created pairs of images
        An array containing information if they are positive or negative.
    '''
    
    # ALEX
    pairs = []
    labels = []
    
    # Defines the range of digits that are in the current dataset from where the pairs are to be created.
    if (test_index == 1):
        digits = [2,3,4,5,6,7]
    if (test_index == 2):
        digits = [0,1,8,9]
    if (test_index == 3):
        digits = [0,1,2,3,4,5,6,7,8,9]
    
    # ALEX
    min_sample = [len(digit_indices[d]) for d in range(len(digits))]
    
    # ALEX
    n = min(min_sample) -1
    
    # ALEX
    for d in range(len(digits)):
        
        # ALEX
        for i in range(n):
            
            # ALEX
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            # ALEX
            pairs += [[x[z1], x[z2]]]
            
            # ALEX
            rand = random.randrange(1, len(digits))
            dn = (d + rand) % len(digits)
            
            # ALEX
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            # ALEX
            labels += [1, 0]
    # ALEX
    return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):
    '''
    For evaluating the prediction accuracy of the model.
    
    ALEX
    
    -args:
        y_true      *DESCRIPTION*
        y_pred      *DESCRIPTION*
        
    -returns:
        *DECRIPTION* 
    '''
    
    # ALEX
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''
    ALEX
    
    -args:
        y_true      *DESCRIPTION*
        y_pred      *DESCRIPTION*
        
    -returns:
        *DECRIPTION* 
    '''
    
    # ALEX
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def siamese_network():
    '''
    Main function to be run for the assignment.
    Will generate data based on the three specified ways of evaluation
    
    Loads a dataset of images from keras called MNIST. Based on these images pairs are created if they are regarded to be the same digit or not. These pairs are used for training and testing a siamese network model that uses a Sequential CNN model as a base network.
    
    The model is trained on data images showing digits in the following set [2,3,4,5,6,7]
    
    The siamese network model is tested and avaluated on 3 different datasets from MNIST:
        ++ The first dataset consists of images with digits in set - set1_digits - [2,3,4,5,6,7] - I.e. the same as the model has been trained on.
        ++ The second dataset consists of images with digits in set - set2_digits - [0,1,8,9] - I.e. digits that are not know to the model.
        ++ The last and third dataset consists of images with digits in set - set3_digits - [0,1,2,3,4,5,6,7,8,9] - I.e. a combination of digits that the model knows AND digits that it does not know.
    '''
    
    # Loads the dataset.
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    
    # Prepocesses the data into 4 different datasets - Input is the image data från MNIST and target is the digit showed in the image.
    (input_trainset, target_trainset), (input_testset1, target_testset1), (input_testset2, target_testset2), (input_testset3, target_testset3) = preprocess_mnist_dataset(x_train, y_train, x_test, y_test)
    
    # Reshape and normalise the input data
    # TRhis is applied to all image data sets after the splitting.
    input_trainset = reshape_convert_input_data(input_trainset)
    input_testset1 = reshape_convert_input_data(input_testset1)
    input_testset2 = reshape_convert_input_data(input_testset2)
    input_testset3 = reshape_convert_input_data(input_testset3)
    
    # Not using all the pictures - For faster debugging 
    #input_trainset = input_trainset[:20000]
    #input_testset1 = input_testset1[:20000]
    #input_testset2 = input_testset2[:20000]
    #input_testset3 = input_testset3[:20000]
    
    # Printing information about the initial datasets before converting into pairs. In order to demonstrate the amount of images that are used for training and testing respectively
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset1: ", input_testset1.shape)
    print("target_testset1: ", target_testset1.shape)
    print("input_testset2: ", input_testset2.shape)
    print("target_testset2: ", target_testset2.shape)
    print("input_testset3: ", input_testset3.shape)
    print("target_testset3: ", target_testset3.shape)

    
    
    # The specific digits that are used for the different sets of image pairs can be found at the top of this document.
    # Creates pairs of images that will be used to train the model with digits in set1_digits
    digit_indices = [np.where(target_trainset == i)[0] for i in set1_digits]
    training_pairs, training_target = create_pairs_set(input_trainset, digit_indices, 1)
    
    # Creates pairs of images that will be used to test the model with digits in set1_digits
    digit_indices = [np.where(target_testset1 == i)[0] for i in set1_digits]
    test_pairs_set1, test_target_set1 = create_pairs_set(input_testset1, digit_indices, 1)
    
    # Creates pairs of images that will be used to test the model with digits in set2_digits
    digit_indices = [np.where(target_testset2 == i)[0] for i in set2_digits]
    test_pairs_set2, test_target_set2 = create_pairs_set(input_testset2, digit_indices, 2)
    
    # Creates pairs of images that will be used to test the model with digits in set3_digits
    digit_indices = [np.where(target_testset3 == i)[0] for i in set3_digits]
    test_pairs_set3, test_target_set3 = create_pairs_set(input_testset3, digit_indices, 3)
    

    #input_shape = (img_rows, img_cols, 1)
    input_shape = input_trainset.shape[1:]
    
    # Shows the shapes of the traning and testing sets for the user. To define how many image pairs that are used for training and testing in the different cases.
    print("--------------------")
    print("training_pairs: ", training_pairs.shape)
    print("training_target: ", training_target.shape)
    print("--------------------")
    print("test_pairs_set1: ", test_pairs_set1.shape)
    print("test_target_set1: ", test_target_set1.shape)
    print("--------------------")
    print("test_pairs_set2: ", test_pairs_set2.shape)
    print("test_target_set2: ", test_target_set2.shape)
    print("--------------------")
    print("test_pairs_set3: ", test_pairs_set3.shape)
    print("test_target_set3: ", test_target_set3.shape)
    print("--------------------")
    
    
    
    # Loops through the creation, training and evalutation of the siamese network model 3 times. For each time there is a new dataset to validate the model on. In order to generate data about validation accuracy and validation loss after each epoch that is run.
    for i in range(3):
        
        # Use a CNN network as the shared network.
        cnn_network_model = build_CNN(input_shape)

        # Initiates inputs with the same amount of slots to keep the image arrays sequences to be used as input data when processing the inputs. 
        image_vector_shape_1 = Input(shape=input_shape)
        image_vector_shape_2 = Input(shape=input_shape)

        # The CNN network model will be shared including weights
        output_cnn_1 = cnn_network_model(image_vector_shape_1)
        output_cnn_2 = cnn_network_model(image_vector_shape_2)

        # Concatenates the two output vectors into one.
        distance = keras.layers.Lambda(euclidean_distance, 
                                       output_shape=eucl_dist_output_shape)([output_cnn_1, output_cnn_2])
        
        # We define a trainable model linking the two different image inputs to the distance between the        processed input by the cnn network.    
        model = Model([image_vector_shape_1, image_vector_shape_2], 
                      distance
                     )
        
        # Depending on what loop index is running different test data is used for validation.
        if (i == 0):
            test_pairs = test_pairs_set1
            test_target = test_target_set1
        if (i == 1):
            test_pairs = test_pairs_set2
            test_target = test_target_set2
        if (i == 2):
            test_pairs = test_pairs_set3
            test_target = test_target_set3
        
        
        # Specifying the optimizer for the netwrok model
        rms = keras.optimizers.RMSprop()
        
        # Compiles the model with the contrastive loss function.
        model.compile(loss=contrastive_loss_function, 
                      optimizer=rms, 
                      metrics=[accuracy])
    
        # Number of epochs is defined in the beginning of the document as a static variable.
        # Validating and printing data using the test data with index i.
        model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_target,
                  batch_size=128,
                  epochs=epochs_siamese,
                  validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_target)
                 )
        
        print('-------------------------------------------------------------------------------')
        print('----------------- THIS DATA WILL PROBABLY NOT BE USED -------------------------')
        print('Final accuracies for the different datasets using validation set number', (i+1), "after a total of", epochs_siamese, "epoch(s).")
        # Compute and print final accuracy, as percentage with 2 decimals, on training and test sets.
        y_pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
        print('* Accuracy on training set: %0.2f%%' % (100 * compute_accuracy(training_target, y_pred)))
        y_pred = model.predict([test_pairs_set1[:, 0], test_pairs_set1[:, 1]])
        print('* Accuracy on test set 1: %0.2f%%' % (100 * compute_accuracy(test_target_set1, y_pred)))
        y_pred = model.predict([test_pairs_set2[:, 0], test_pairs_set2[:, 1]])
        print('* Accuracy on test set 2: %0.2f%%' % (100 * compute_accuracy(test_target_set2, y_pred)))
        y_pred = model.predict([test_pairs_set3[:, 0], test_pairs_set3[:, 1]])
        print('* Accuracy on test set 3: %0.2f%%' % (100 * compute_accuracy(test_target_set3, y_pred)))
        print('-------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------')
    

    
'''
---------------------------------- END OF DOCUMENT ----------------------------------
'''