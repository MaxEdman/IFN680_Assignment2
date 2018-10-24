
import numpy as np
from tensorflow import keras

# Imports required modules from the Keras Functional API
import keras
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model


# Defining a set of static variables that are used throughout the defined functions.
batch_size = 128
epochs_cnn = 1
epochs_siamese = 1
# In the MNIST set the number of classes is 10.
num_classes = 10





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
    
    return (x_train, y_train), (x_test, y_test)





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
    tnt_dataset = image_dataset[train_and_test_mask,:,:]
    #print("tnt_dataset shape: {0}", tnt_dataset.shape)
    tnt_target = target_dataset[train_and_test_mask]
    #print("tnt_target shape: {0}", tnt_target.shape)
    
    # Initiating two new arrays with data and target based on the remaining entries in the array. I.e. the negative of the mask. These will ONLY be used for testing.
    only_test_dataset = image_dataset[~train_and_test_mask,:,:]
    #print("only_test_dataset shape: {0}", only_test_dataset.shape)
    only_test_target = target_dataset[~train_and_test_mask]
    #print("only_test_target shape: {0}", only_test_target.shape)
    
    # Import module for splitting datasets.
    from sklearn.model_selection import train_test_split
    # Splits the dataset that are used for both train and test into respective sets. With test size of 20% as stated in the implementation hints "Keep 80% of the [2,3,4,5,6,7] digits for training (and 20% for testing)."
    data_train, data_test, target_train, target_test = train_test_split(tnt_dataset, tnt_target, test_size=0.20)
    
    # Concatenates the data that should be used for testing.
    final_test_data = np.concatenate([data_test, only_test_dataset])
    final_test_target = np.concatenate([target_test, only_test_target])
    
    # Returns the data to be used in training (80% of the digits [2,3,4,5,6,7]) and testing (the rest).
    return data_train, target_train, final_test_data, final_test_target






def contrastive_loss_function(y_true, y_pred):
    
    # The margin m > 0 determines how far the embeddings of a negative pair should be pushed apart.
    m = 2 # Might need to be changed and evaluated for what value the siamese network performs best.
    
    # Calclulates the euclidian distance
    e_dist = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    
    return (abs(y_true - 1)) * ((e_dist**2) / 2) + (y_true * ((K.maximum(float(0), (m - e_dist))**2) / 2))
    
    
    
    
def reshape_convert_input_data(x_train, x_test):
    
    # Code adapted from practical 7 when training a CNN 
    img_rows, img_cols = x_train.shape[1:3]
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    # convert to float32 and rescale between 0 and 1.
    # All the data is an RGB number between 0 and 255. Which is why dividing the vectors with 255 rescales the input data between 0 and 1.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return x_train, x_test
    
    
    
    
    
def build_CNN(input_shape):
    '''
    Build a CNN model to be used as a shared network in the siamese network model.
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
    
    # Calls function to retreive the locally saved dataset.
    #(x_train, y_train), (x_test, y_test) = load_mnist_dataset()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Split dataset
    x_set, y_set = split_dataset(x_train, y_train, x_test, y_test)
    x_train, x_test, x_test_unknown = x_set
    y_set1, y_set2, y_set3 = y_set
    
    img_rows, img_cols = x_train.shape[1:3]
    input_shape = (img_rows, img_cols, 1)
    
    x_set1 = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_set2 = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_set3 = x_test_unknown.reshape(x_test_unknown.shape[0], img_rows, img_cols, 1)
    x_set1, x_set2, x_set3 = x_set1.astype('float32'), x_set2.astype('float32'), x_set3.astype('float32')
    x_set1 /= 255
    x_set2 /= 255
    x_set3 /= 255

    # Create training pairs
    digit_idx = [np.where(y_set1 == i)[0] for i in range(num_classes)]
    siamese_pairs_set1, siamese_set1_label = create_pairs(x_set1, digit_idx)

    # Create test pairs
    digit_idx = [np.where(y_set2 == i)[0] for i in range(num_classes)]
    siamese_pairs_set2, siamese_set2_label  = create_pairs(x_set2, digit_idx)

    # Create unknown test pairs
    digit_idx = [np.where(y_set3 == i)[0] for i in range(num_classes)]
    siamese_pairs_set3, siamese_set3_label = create_pairs(x_set3, digit_idx)
    
    # Use a CNN network as the shared network.
    cnn_network_model = build_CNN(input_shape)

    image_vector_shape_a = Input(shape=input_shape)
    image_vector_shape_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
    output_cnn_a = cnn_network_model(image_vector_shape_a)
    output_cnn_b = cnn_network_model(image_vector_shape_b)
    
    # Concatenates the two output vectors into one.
    merged_output = keras.layers.concatenate([output_cnn_a, output_cnn_b])

    # And add a logistic regression on top
    # WHY DO WE DO THIS? AAAH. DON'T GET IT!
    predictions = Dense(1, activation='sigmoid')(merged_output)
    
    # We define a trainable model linking the two different image inputs to the predictions
    model = Model(inputs=[image_vector_shape_a, image_vector_shape_b],
                  outputs=predictions)
                  
    # 
    model.compile(optimizer='rmsprop', loss=contrastive_loss_function)
    #model.compile(loss=contrastive_loss_function, optimizer=keras.optimizers.Adadelta())
    

    # This is where we need to use the negative and positive pairs from the images based on the sequential classes as input along with the labels.
    # Number of epochs is defined in the beginning of the document as a static variable.
    
    model.fit([siamese_pairs_set1[:,0], siamese_pairs_set1[:,1]], 
              siamese_set1_label,
              batch_size=batch_size,
              epochs=epochs_siamese)
    

    
    # How do we test the accuracy on the model ? The Keras API does not say anything about it.
    #score = model.evaluate(x=te1_pairs, y=te1_y, batch_size=128, verbose=1)
    #score = model.evaluate(x=[te2_pairs[:, 0], te2_pairs[:, 1]], y=te2_y, batch_size=128, verbose=1)
    #score = model.evaluate([input_test_pairs[:,0], input_test_pairs[:,1]], test_labels, verbose=True)
    
    #print('Test loss for Siamese network:', score[0])
    print('Test accuracy for Siamese network:', score[1])
    
    
    
    
    
    
    
    
    
    
    
def create_neg_pos_pairs(input_data, indices):
    '''
    This function needs to be further understood. 
    '''
    import random
    
    pairs = []
    labels = []
    n = min([len(indices[d]) for d in range(2,8)]) - 1
    for d in range(2,8):
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
    
    #return x_train_pairs, labels
    '''
        Creates an array of positive and negative pairs combined with their labels. If the two images used as input is considered to be from the same eqivalence class then they are considered a positive pair. If they are not, they are considered a negative pair.
    '''
    
def create_pairs2(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    import random
    
    n = min([len(digit_indices[d]) for d in range(2,8)]) - 1
    for d in range(2,8):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, len(digit_indices))
            dn = (d + inc) % len(digit_indices)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def create_simplistic_base_network(input_shape):
    '''
    Base network to be shared (eq. to feature extraction).
    '''
    seq = keras.models.Sequential()
    seq.add(keras.layers.Dense(128, input_shape=input_shape, activation='relu'))
    seq.add(keras.layers.Dropout(0.1))
    seq.add(keras.layers.Dense(128, activation='relu'))
    seq.add(keras.layers.Dropout(0.1))
    seq.add(keras.layers.Dense(128, activation='relu'))
    return seq
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''

    _input = Input(shape=input_shape)
    x = Flatten()(_input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(_input, x)
    
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    import random
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

def split_dataset(x_tr,y_tr,x_t,y_t): # Split dataset into training and test sets using given integers
    # Combine datasets to be split based on integers
    X = np.concatenate((x_tr,x_t))
    Y = np.concatenate((y_tr,y_t))
    
    tr_ints , t_ints = [2,3,4,5,6,7], [0,1,8,9]
    
    # Creates a boolean mask for each set
    tr_set = [ x in tr_ints for x in Y ]
    t_set = [ x in t_ints for x in Y ]
    
    x_tr, x_t = X[tr_set], X[t_set]
    y_tr, y_t = Y[tr_set], Y[t_set]

    # Split 80% to training and 20% to test
    split = int(len(x_tr) * 0.8)    
    x_t2, x_tr = x_tr[split::], x_tr[:split:]
    y_t2, y_tr = y_tr[split::], y_tr[:split:]

    return ((x_tr, x_t, x_t2)), ((y_tr, y_t, y_t2))

def create_pairs3(xlist, digit_idx):
    #Initialise lists
    pairs = []
    labels = []
    
    digit_len = [len(digit_idx[d]) for d in range(num_classes)] # Get the number of items for each digit/class
    n = min(digit_len) - 1 # Find the length of the smallest set of digits
    
    for d in range(num_classes):
        for i in range(n):
            # Assign positive pair
            pos_idx1, pos_idx2 = digit_idx[d][i], digit_idx[d][i+1]
            pos1, pos2 = xlist[pos_idx1], xlist[pos_idx2]
            pairs += [[pos1,pos2]]
            
            # Assign a random digit that is not the original digit
            rand_d = random.randrange(1,num_classes)
            rand_d = (d + rand_d) % num_classes
            
            # Assign negative pair
            neg_idx1, neg_idx2 = digit_idx[d][i], digit_idx[rand_d][i]
            neg1, neg2 = xlist[neg_idx1], xlist[neg_idx2]
            pairs += [[neg1,neg2]]
            
            # Assign labels for positive and negative pair
            labels += [1,0]
            
    return np.array(pairs), np.array(labels)


def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def euclidean_distance(vec_2d):
    x, y = vec_2d
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return result