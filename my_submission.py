
import numpy as np
from tensorflow import keras
import keras.backend as K

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
    
# Function to implement the preprocessing steps of:
#   Split the dataset such that ...
#   ◦ the digits in [2,3,4,5,6,7] are used for training and testing
#   ◦ the digits in [0,1,8,9] are only used for testing. None of these digits should be used during training.
def preprocess_mnist_dataset(x_train, y_train, x_test, y_test):
    
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
    
    
    '''
    # If y_true == 0 it denotes that the images are deemed from the same equivlaence class. A positive pair. Hence, one function for contrastive loss distance is used.
    if (y_true == 0):
        return ((e_dist**2) / 2)
    
    # If y_true == 1 then the pair of images is a negative pair, and another function for contrastive loss distance is to be used.
    elif (y_true == 1):
        return ((K.maximum(0, (m - e_dist))**2) / 2)
    
    else:
        print("Contrastive loss function does not apply to either y_true == 0 nor y_true == 1")
        print("y_true: {0}", y_true)
        return
    '''
    
def build_CNN(x_train, y_train, x_test, y_test):
    '''
    Build, train and evaluate a CNN on the mnist dataset
    
    '''
    # Code adapted from practical 7 when training a CNN
    
    img_rows, img_cols = x_train.shape[1:3]
    num_classes = len(np.unique(y_test))
    print("num_classes: ", num_classes)
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    batch_size = 128
    epochs = 12

    #    epochs = 3 # debugging code
    #    x_train = x_train[:8000]
    #    y_train = y_train[:8000]


    # convert to float32 and rescale between 0 and 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    #
    # convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
              
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    '''
    img_rows, img_cols = x_train.shape[1:3]
    num_classes = len(np.unique(y_test))
    print("num_classes: ", num_classes)
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    batch_size = 128
    epochs = 12

    '''
    #epochs = 3 # debugging code
    #x_train = x_train[:8000]
    #y_train = y_train[:8000]
    '''


    # convert to float32 and rescale between 0 and 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(loss=contrastive_loss_function,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
              
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return
    '''

def siamese_network(x_train, y_train, x_test, y_test):
    
    # Builds the siamese network
    shared_network = build_CNN(x_train, y_train, x_test, y_test)
