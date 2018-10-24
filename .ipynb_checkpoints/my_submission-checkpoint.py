
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
    
    
    
    
    
def build_CNN(x_train, y_train, x_test, y_test):
    '''
    Build, train and evaluate a CNN on the MNIST dataset
    '''
    
    # Is set static to 10 in the top of the document.
    print("num_classes: ", num_classes)
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train, x_test = reshape_convert_input_data(x_train, x_test)
    
    # Defines the shape of the input data for the CNN. Should be (28, 28, 1)
    # To be used by one of the added layers to the CNN model.
    img_rows, img_cols = x_train.shape[1:3]
    input_shape = (img_rows, img_cols, 1)
    
    #print('x_train shape:', x_train.shape)
    #print('input_shape:', input_shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    
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
    
    
    '''
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    #print("x_test shape: ", x_test.shape)
    #print("y_test shape: ", y_test.shape)
    # Fits the model to the training set.
    # Number of epochs are definied in the top of this document.
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs_cnn,
              verbose=True,
              validation_data=(x_test, y_test))
              
    score = model.evaluate(x_test, y_test, verbose=True)
    
    print('Test loss for CNN as shared network:', score[0])
    print('Test accuracy for CNN as shared network:', score[1])
    '''
    
    return model
    



def siamese_network(x_train, y_train, x_test, y_test):
    '''
    Main function to be run for the assignment.
    '''
    
    
    
    
    (x1_train, y1_train), (x1_test, y1_test) = load_mnist_dataset()
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    img_rows, img_cols = x1_train.shape[1:3]
    x1_train = x1_train.reshape(x1_train.shape[0], img_rows, img_cols, 1)
    x1_test = x1_test.reshape(x1_test.shape[0], img_rows, img_cols, 1)
    
    print("x1_train: ", x1_train.shape)
    print("y1_train: ", y1_train.shape)
    print("x1_test: ", x1_test.shape)
    print("y1_test: ", y1_test.shape)
    
    
    
    (x2_train, y2_train), (x2_test, y2_test) = keras.datasets.mnist.load_data()
    img_rows, img_cols = x2_train.shape[1:3]
    x2_train = x2_train.reshape(x2_train.shape[0], img_rows, img_cols, 1)
    x2_test = x2_test.reshape(x2_test.shape[0], img_rows, img_cols, 1)
    
    print("x2_train: ", x2_train.shape)
    print("y2_train: ", y2_train.shape)
    print("x2_test: ", x2_test.shape)
    print("y2_test: ", y2_test.shape)
    
    
    
    
    
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
    
    
    # create training positive and negative pairs
    digit_indices1 = [np.where(y1_train == i)[0] for i in range(2,8)]
    print("digit_indices1.len: ", len(digit_indices1))
    print("range(num_classes): ", range(2,8))
    tr1_pairs, tr1_y = create_neg_pos_pairs(x1_train, digit_indices1)
    #tr_pairs, tr_y = create_neg_pos_pairs(x_train, digit_indices)

    digit_indices1 = [np.where(y1_test == i)[0] for i in range(num_classes)]
    #print("digit_indices 2: ", digit_indices)
    te1_pairs, te1_y = create_pairs(x1_test, digit_indices1)
    #te_pairs, te_y = create_neg_pos_pairs(x_test, digit_indices)
    
    
    # create training+test positive and negative pairs
    digit_indices2 = [np.where(y2_train == i)[0] for i in range(num_classes)]
    print("digit_indices2: ", len(digit_indices2))
    #print("digit_indices: ", digit_indices)
    tr2_pairs, tr2_y = create_pairs(x2_train, digit_indices2)
    #tr_pairs, tr_y = create_neg_pos_pairs(x_train, digit_indices)

    digit_indices2 = [np.where(y2_test == i)[0] for i in range(num_classes)]
    #print("digit_indices 2: ", digit_indices)
    te2_pairs, te2_y = create_pairs(x2_test, digit_indices2)
    #te_pairs, te_y = create_neg_pos_pairs(x_test, digit_indices)
    
    
    
    print("tr1_pairs: ", tr1_pairs.shape)
    print("tr1_y: ", tr1_y.shape)
    print("te1_pairs: ", te1_pairs.shape)
    print("te1_y: ", te1_y.shape)
    
    print("tr2_pairs: ", tr2_pairs.shape)
    print("tr2_y: ", tr2_y.shape)
    print("te2_pairs: ", te2_pairs.shape)
    print("te2_y: ", te2_y.shape)
    
    
    '''
    
    indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    input_train_pairs, train_labels = create_pairs(x_test, indices)
    #input_train_pairs, train_labels = create_neg_pos_pairs(x_train, indices)
    
    indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    input_test_pairs, test_labels = create_pairs(x_test, indices)
    #input_test_pairs, test_labels = create_neg_pos_pairs(x_test, indices)
    '''
    '''
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    pairs_train, label_train = reshape_convert_input_data(x_train, y_train)
    pairs_test, label_test = reshape_convert_input_data(x_test, y_test)
    '''
    
    '''
    # Use a CNN network as the shared network.
    cnn_network_model = build_CNN(x_train, y_train, x_test, y_test)
    
    base_network = create_simplistic_base_network(input_shape)
    
    print("x_train.shape[1:]: ", x_train.shape[1:])
    input_shape = x_train.shape[1:]

    # Initiates inputs with the same amount of slots to keep the image arrays sequences to be used as input data when processing the inputs. 
    image_vector_shape_a = Input(shape=input_shape)
    image_vector_shape_b = Input(shape=input_shape)
    
    print("input_shape: ", input_shape)
    print("image_vector_shape_a:", image_vector_shape_a)
    print("cnn_network_model:", cnn_network_model)
    
    # The CNN network model will be shared including weights
    #output_cnn_a = cnn_network_model(image_vector_shape_a)
    #output_cnn_b = cnn_network_model(image_vector_shape_b)
    output_cnn_a = base_network(image_vector_shape_a)
    output_cnn_b = base_network(image_vector_shape_b)
    
    
    
    '''
    

    print("input_shape1:", input_shape1)
    print("input_shape2:", input_shape2)
    
    # Use a CNN network as the shared network.
    #cnn_network_model = build_CNN(x_train, y_train, x_test, y_test)
    #cnn_network_model1 = build_CNN(x1_train, y1_train, x1_test, y1_test)
    cnn_network_model2 = build_CNN(x2_train, y2_train, x2_test, y2_test)
    
    
    # network definition
    #base_network1 = create_base_network(input_shape1)
    #base_network2 = create_base_network(input_shape2)
    
    input_shape_test = (img_rows, img_cols, 1)
    
    #input_a = Input(shape=input_shape)
    #input_b = Input(shape=input_shape)
    #image_vector_shape_a = Input(shape=input_shape1)
    #image_vector_shape_b = Input(shape=input_shape1)
    image_vector_shape_a = Input(shape=input_shape_test)
    image_vector_shape_b = Input(shape=input_shape_test)
    
    print("image_vector_shape_a:", image_vector_shape_a)
    print("image_vector_shape_a:", image_vector_shape_b)
    #print("base_network:", base_network)

    # because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
    #output_cnn_a = cnn_network_model(image_vector_shape_a)
    #output_cnn_b = cnn_network_model(image_vector_shape_b)
    
    output_cnn_a = cnn_network_model2(image_vector_shape_a)
    output_cnn_b = cnn_network_model2(image_vector_shape_b)
    
    #output_cnn_a = base_network1(image_vector_shape_b)
    #output_cnn_b = base_network1(image_vector_shape_b)
    
    
    
    
    
    
    
    # Concatenates the two output vectors into one.
    merged_output = keras.layers.concatenate([output_cnn_a, output_cnn_b])
    
    # And add a logistic regression on top
    # WHY DO WE DO THIS? AAAH. DON'T GET IT!
    predictions = Dense(1, activation='sigmoid')(merged_output)
    
    # We define a trainable model linking the two different image inputs to the predictions
    model = Model(inputs=[image_vector_shape_a, image_vector_shape_b], outputs=predictions)
    
    # 
    model.compile(optimizer='rmsprop',
                  loss=contrastive_loss_function)
    
    
    print("tr1_pairs: ", tr1_pairs.shape)
    print("tr1_y: ", tr1_y.shape)
    print("te1_pairs: ", te1_pairs.shape)
    print("te1_y: ", te1_y.shape)
    
    print("tr2_pairs: ", tr2_pairs.shape)
    print("tr2_y: ", tr2_y.shape)
    print("te2_pairs: ", te2_pairs.shape)
    print("te2_y: ", te2_y.shape)
    
    
    # This is where we need to use the negative and positive pairs from the images based on the sequential classes as input along with the labels.
    # Number of epochs is defined in the beginning of the document as a static variable.
    
    '''
    model.fit([input_train_pairs[:,0], 
               input_train_pairs[:,1]], 
              train_labels,
              epochs=epochs_siamese)
    
    
    model.fit([tr1_pairs[:, 0], tr1_pairs[:, 1]], tr1_y,
              batch_size=128,
              epochs=epochs_siamese,
              validation_data=([te1_pairs[:, 0], te1_pairs[:, 1]], te1_y))
    
    
    '''
    model.fit([tr2_pairs[:, 0], tr2_pairs[:, 1]], tr2_y,
              batch_size=128,
              epochs=epochs_siamese,
              validation_data=([te2_pairs[:, 0], te2_pairs[:, 1]], te2_y))
    
    
    
    
    
    # How do we test the accuracy on the model ? The Keras API does not say anything about it.
    #score = model.evaluate(x=te1_pairs, y=te1_y, batch_size=128, verbose=1)
    score = model.evaluate(x=[tr2_pairs[:, 0], tr2_pairs[:, 1]], y=te2_y, batch_size=128, verbose=1)
    #score = model.evaluate(input_test_pairs, test_labels, verbose=True)
    
    #print('Test loss for Siamese network:', score[0])
    #print('Test accuracy for Siamese network:', score[1])
    
    
    
def create_neg_pos_pairs(input_data, indices):
    '''
    This function needs to be further understood. 
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
    
    #return x_train_pairs, labels
    '''
        Creates an array of positive and negative pairs combined with their labels. If the two images used as input is considered to be from the same eqivalence class then they are considered a positive pair. If they are not, they are considered a negative pair.
    '''
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
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