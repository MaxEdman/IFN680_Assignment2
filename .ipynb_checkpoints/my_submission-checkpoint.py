
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
epochs_siamese = 1
# In the MNIST set the number of classes is 10.
#num_train_classes = 6
num_classes = 10

train_digits = [2,3,4,5,6,7]




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
    
    
    
def reshape_convert_input_data(input_data):
    
    # Code adapted from practical 7 when training a CNN 
    img_rows, img_cols = input_data.shape[1:3]
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
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
    (x_train, y_train), (x_test, y_test), (input_testset2, target_testset2), (input_testset3, target_testset3) = preprocess_mnist_dataset(x_train, y_train, x_test, y_test)
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = reshape_convert_input_data(x_train)
    x_test = reshape_convert_input_data(x_test)
    
    
    # For debugging
    #x_train = x_train[:8000]
    #y_train = y_train[:8000]
    
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    
    
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
    
    
    
    '''
    print("x2_train: ", x2_train.shape)
    print("y2_train: ", y2_train.shape)
    print("x2_test: ", x2_test.shape)
    print("y2_test: ", y2_test.shape)
    
    print("tr1_pairs: ", tr1_pairs.shape)
    print("tr1_y: ", tr1_y.shape)
    print("te1_pairs: ", te1_pairs.shape)
    print("te1_y: ", te1_y.shape)
    
    print("tr2_pairs: ", tr2_pairs.shape)
    print("tr2_y: ", tr2_y.shape)
    print("te2_pairs: ", te2_pairs.shape)
    print("te2_y: ", te2_y.shape)
    

    print("input_shape1:", input_shape1)
    print("input_shape2:", input_shape2)
    '''
    '''
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    '''
    
    print("train_digits", train_digits)
    # create training positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in train_digits]
    tr_pairs, tr_y = create_pairs_set1(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in train_digits]
    te_pairs, te_y = create_pairs_set1(x_test, digit_indices)
    

    #input_shape = (img_rows, img_cols, 1)
    input_shape = x_train.shape[1:]
    
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
    
    '''
    # 
    model.compile(optimizer='rmsprop',
                  loss=contrastive_loss_function)
    '''
    
    print("tr_pairs: ", tr_pairs.shape)
    print("tr_y: ", tr_y.shape)
    print("te_pairs: ", te_pairs.shape)
    print("te_y: ", te_y.shape)
    
    # Training the model
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss_function, optimizer=rms, metrics=[accuracy])
    #model.compile(loss=contrastive_loss_function, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs_siamese)
    
    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    '''
    print("tr1_pairs: ", tr1_pairs.shape)
    print("tr1_y: ", tr1_y.shape)
    print("te1_pairs: ", te1_pairs.shape)
    print("te1_y: ", te1_y.shape)
    
    print("tr2_pairs: ", tr2_pairs.shape)
    print("tr2_y: ", tr2_y.shape)
    print("te2_pairs: ", te2_pairs.shape)
    print("te2_y: ", te2_y.shape)
    '''
    
    # This is where we need to use the negative and positive pairs from the images based on the sequential classes as input along with the labels.
    # Number of epochs is defined in the beginning of the document as a static variable.
    
    
    
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)



def create_pairs_set1(x, digit_indices):
    
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
    
def create_neg_pos_pairs(input_data, indices):
    '''
    Creates an array of positive and negative pairs combined with their labels. If the two images used as input is considered to be from the same eqivalence class then they are considered a positive pair. If they are not, they are considered a negative pair.
    '''
    import random
    
    pairs = []
    labels = []
    n = min([len(indices[d]) for d in range(len(indices))]) - 1
    for d in range(len(indices)):
        for i in range(n):
            j1 = indices[d][i]
            j2 = indices[d][i + 1]
            pairs.append([[input_data[j1], input_data[j2]]])
            
            # Don't know if this should be in the range 2-8 (the numbers that are in the set for training) or if it should be the length of indices.
            inc = random.randrange(len(indices))
            dn = (d + inc) % len(indices)

            k1 = indices[d][i]
            k2 = indices[dn][i]
            pairs += [[input_data[k1], input_data[k2]]]
            
            labels += [1, 0]
            
    return np.array(pairs), np.array(labels)
    
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    
    
