
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import random

# Imports required modules from the Keras Functional API
import keras
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model


# Defining a set of static variables that are used throughout the defined functions.
batch_size = 128
epochs_siamese = 10
# In the MNIST set the number of classes is 10.
#num_train_classes = 6
num_classes = 10

# Defining ranges with for different test sets based on the target label of the different pictures.
set1_digits = [2,3,4,5,6,7]
set2_digits = [0,1,8,9]
set3_digits = [0,1,2,3,4,5,6,7,8,9]




# Defining function to save the dataset to be used in Assignment #2 in Unit IFN680 @ QUT
def save_mnist_dataset():
    from tensorflow.contrib import keras
    from keras.datasets import mnist
    
    # Loads the dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Saves the dataset locally as a .npz file.
    np.savez('mnist_dataset.npz',
             x_train = x_train,
             y_train = y_train,
             x_test = x_test,
             y_test = y_test)

# Function to load the mnist dataset    
def load_mnist_dataset():
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
    '''
    
    # Using numpy module to concatenate train and test data into their own datasets. Respectively for x and y
    image_dataset = np.concatenate([x_train, x_test])
    target_dataset = np.concatenate([y_train, y_test])
    
    # Creates a mask with all numbers that should be used for both training and testing. I.e. where target dataset is any of the following numbers [2,3,4,5,6,7]. Will be further called tnt_dataset
    train_and_test_mask = np.logical_and(target_dataset>1, target_dataset<8)
    #print("train_and_test_mask shape: {0}", train_and_test_mask.shape)
    #print("image_dataset shape: {0}", image_dataset.shape)
    
    # Initiating two new arrays with data and target based on previous created mask. These will be used for both testing and traning.
    #tnt_dataset = np.ma.array(image_dataset, mask=train_and_test_mask)
    test_and_train_dataset = image_dataset[train_and_test_mask,:,:]
    #print("tnt_dataset shape: {0}", tnt_dataset.shape)
    test_and_train_target = target_dataset[train_and_test_mask]
    #print("tnt_target shape: {0}", tnt_target.shape)
    
    # Initiating two new arrays with data and target based on the remaining entries in the array. I.e. the negative of the mask. These will ONLY be used for testing.
    only_test_dataset = image_dataset[~train_and_test_mask,:,:]
    #print("only_test_dataset shape: {0}", only_test_dataset.shape)
    only_test_target = target_dataset[~train_and_test_mask]
    #print("only_test_target shape: {0}", only_test_target.shape)
    
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





def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss_function(y_true, y_pred):
    
    # The margin m > 0 determines how far the embeddings of a negative pair should be pushed apart.
    m = 2 # Might need to be changed and evaluated for what value the siamese network performs best.
    
    # Calclulates the euclidian distance
    e_dist = euclidean_distance((y_pred, y_true))
    
    return (abs(y_true - 1)) * ((e_dist**2) / 2) + (y_true * ((K.maximum(float(0), (m - e_dist))**2) / 2))


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss 
    '''
    m = 2 # margin
    euc_dist = euclidean_distance((y_pred, y_true))
    
    #sqaure_pred = K.square(euc_dist)/2
    sqaure_pred = K.square(y_pred)
    
    #margin_square = K.square(K.maximum(float(0), m - euc_dist))/2
    margin_square = K.square(K.maximum(m - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
    #return K.mean(abs(y_true - 1) * sqaure_pred + (y_true * margin_square))
    
    
def reshape_convert_input_data(input_data):
    
    # Code adapted from practical 7 when training a CNN 
    img_rows, img_cols = input_data.shape[1:3]
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    #input_data = input_data.reshape(input_data.shape[0], img_rows, img_cols)
    input_data = input_data.reshape(input_data.shape[0], img_rows, img_cols, 1)
    
    # convert to float32 and rescale between 0 and 1.
    # All the data is an RGB number between 0 and 255. Which is why dividing the vectors with 255 rescales the input data between 0 and 1.
    input_data = input_data.astype('float32')
    input_data /= 255
    
    return input_data
    
    
    
    
    
def build_CNN(input_shape):
    '''
    Build a CNN model to be used as a shared network in the siamese network model.
    Mainly copied from the CNN practical during week 7.
    '''
    
    cnn_model = keras.models.Sequential()
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
    
    return cnn_model
    



def siamese_network():
    '''
    Main function to be run for the assignment.
    '''
    
    # Loads the dataset.
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    #mnist.load_data()
    (input_trainset, target_trainset), (input_testset1, target_testset1), (input_testset2, target_testset2), (input_testset3, target_testset3) = preprocess_mnist_dataset(x_train, y_train, x_test, y_test)
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    input_trainset = reshape_convert_input_data(input_trainset)
    input_testset1 = reshape_convert_input_data(input_testset1)
    input_testset2 = reshape_convert_input_data(input_testset2)
    input_testset3 = reshape_convert_input_data(input_testset3)
    
    
    # For debugging
    #x_train = x_train[:8000]
    #y_train = y_train[:8000]
    
    print("input_trainset: ", input_trainset.shape)
    print("target_trainset: ", target_trainset.shape)
    print("input_testset1: ", input_testset1.shape)
    print("target_testset1: ", target_testset1.shape)
    print("input_testset2: ", input_testset2.shape)
    print("target_testset2: ", target_testset2.shape)
    print("input_testset3: ", input_testset3.shape)
    print("target_testset3: ", target_testset3.shape)
    
    
    '''
    x1_train = x1_train.astype('float32')
    x1_test = x1_test.astype('float32')
    x1_train /= 255
    x1_test /= 255
    input_shape1 = x1_train.shape[1:]
    
    
    x2_train = x2_train.astype('float32')
    x2_test = x2_test.astype('float32')
    x2_train /= 255
    x2_test /= 255
    input_shape2 = x2_train.shape[1:]
    '''
    
    
    
    print("set1_digits", set1_digits)
    # create training positive and negative pairs
    digit_indices = [np.where(target_trainset == i)[0] for i in set1_digits]
    training_pairs, training_target = create_pairs_set(input_trainset, digit_indices, 1)

    digit_indices = [np.where(target_testset1 == i)[0] for i in set1_digits]
    test_pairs_set1, test_target_set1 = create_pairs_set(input_testset1, digit_indices, 1)
    
    digit_indices = [np.where(target_testset2 == i)[0] for i in set2_digits]
    test_pairs_set2, test_target_set2 = create_pairs_set(input_testset2, digit_indices, 2)
    
    digit_indices = [np.where(target_testset3 == i)[0] for i in set3_digits]
    test_pairs_set3, test_target_set3 = create_pairs_set(input_testset3, digit_indices, 3)
    

    #input_shape = (img_rows, img_cols, 1)
    input_shape = input_trainset.shape[1:]
    
    # Use a CNN network as the shared network.
    cnn_network_model = build_CNN(input_shape)

    # Initiates inputs with the same amount of slots to keep the image arrays sequences to be used as input data when processing the inputs. 
    image_vector_shape_a = Input(shape=input_shape)
    image_vector_shape_b = Input(shape=input_shape)
    
    # The CNN network model will be shared including weights
    output_cnn_a = cnn_network_model(image_vector_shape_a)
    output_cnn_b = cnn_network_model(image_vector_shape_b)

    # Concatenates the two output vectors into one.
    distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([output_cnn_a, output_cnn_b])
    
    # And add a logistic regression on top
    # WHY DO WE DO THIS? AAAH. DON'T GET IT! Shouldn't be in here?
    #predictions = Dense(1, activation='sigmoid')(merged_output)
    
    # We define a trainable model linking the two different image inputs to the distance between the processed input by the cnn network.    
    model = Model([image_vector_shape_a, image_vector_shape_b], distance)
    

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
    
    
    # Training the model
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    #model.compile(loss=contrastive_loss_function, optimizer=rms)
    # Only validating on the data used to fit the model.
    model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_target,
              batch_size=128,
              epochs=epochs_siamese,
              validation_data=([test_pairs_set1[:, 0], test_pairs_set1[:, 1]], test_target_set1))
    
    # compute final accuracy on training and test sets
    y_pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
    training_acc = compute_accuracy(training_target, y_pred)
    print('* Accuracy on training set: %0.2f%%' % (100 * training_acc))
    
    y_pred = model.predict([test_pairs_set1[:, 0], test_pairs_set1[:, 1]])
    test_acc_set1 = compute_accuracy(test_target_set1, y_pred)
    print('* Accuracy on test set 1: %0.2f%%' % (100 * test_acc_set1))
    
    y_pred = model.predict([test_pairs_set2[:, 0], test_pairs_set2[:, 1]])
    test_acc_set2 = compute_accuracy(test_target_set2, y_pred)
    print('* Accuracy on test set 2: %0.2f%%' % (100 * test_acc_set2))
    
    y_pred = model.predict([test_pairs_set3[:, 0], test_pairs_set3[:, 1]])
    test_acc_set3 = compute_accuracy(test_target_set3, y_pred)
    print('* Accuracy on test set 3: %0.2f%%' % (100 * test_acc_set3))

    
    # This is where we need to use the negative and positive pairs from the images based on the sequential classes as input along with the labels.
    # Number of epochs is defined in the beginning of the document as a static variable.
    



    
    
def create_pairs_set(x, digit_indices, test_index):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    
    Creates an array of positive and negative pairs combined with their labels. If the two images used as input       is considered to be from the same eqivalence class then they are considered a positive pair. If they are not,     they are considered a negative pair.
    
    '''
    
    # Create empty list of pairs and labels to be appended
    pairs = []
    labels = []
    
    if (test_index == 1):
        digits = [2,3,4,5,6,7]
    if (test_index == 2):
        digits = [0,1,8,9]
    if (test_index == 3):
        digits = [0,1,2,3,4,5,6,7,8,9]
    
    # calculate the min number of training sample of each digit in training set
    min_sample = [len(digit_indices[d]) for d in range(len(digits))]
    
    # calculate the number of pairs to be created
    n = min(min_sample) -1
    
    # Looping over each digits in the train_digits
    for d in range(len(digits)):
        
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
            rand = random.randrange(1, len(digits))
            dn = (d + rand) % len(digits)
            
            # Use the dn and d to create a pair of different digits 
            # the append it to the pair list
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            # Append the corresponding label value for the true and false pairs of image created
            labels += [1, 0]
    
    return np.array(pairs), np.array(labels)



### For evaluating the prediction accuracy of the model.    
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
