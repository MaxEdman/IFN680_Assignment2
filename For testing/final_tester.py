import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras.models import Model

"""
Split the dataset so that training occurs on digits [2,3,4,5,6,7] and testing occurs accross all digits
"""
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
"""
Create pairs from dataset that alternate between positive and negative
"""
def create_pairs(xlist, digit_idx):
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

"""
Create the base model for siamese network, this is based on CNN architecture
"""
def base_model(input_shape, num_classes):
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

    return model

"""
Define functions to compute euclidean distance.
The eucledian distance can be computed by square all the distance between x and y, then square it (power of 2). 
Lastly, the distance is the squareroot of the sum all of results. 
:param: 2D vector
:return enclidean distance
"""
def euclidean_distance(vec_2d):
    x, y = vec_2d
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return result

"""
Define functions to convert the shape of euclidean distance function.
This will be used in to define the output_shape of the Lambda layer
:params: shapes (2D/x and y)
:return: tuples(x,1)
"""
def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

"""
Generate accuracy for siamese net.
The threshold is set to 0.5, if the distance predicted is more than threshold, it will be counted as 0 (= False).
In other words, if the distance between pair is close enough, it will be consider them as identical pair.
Finally the prediction is compared to the ground truth
:params y_ground_truth: ground truth of the data
:params y_pred: prediction result from the model
:return: accuracy (since the data are either 0 or 1 (true or false), we can use mean function of comparison to compute accuracy)
"""
def compute_accuracy(y_ground_truth, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_ground_truth)


"""
Main code start 
"""
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

"""
1. Get input size to be used later in defining Sequencial model
"""
# Define image dimension (rows and cols) and number of classes
img_rows, img_cols = 28, 28
num_classes = 10

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

left_input = keras.layers.Input(shape=input_shape)
right_input = keras.layers.Input(shape=input_shape)
"""
2. Preprocess dataset
"""
# Split dataset
x_set, y_set = split_dataset(x_train, y_train, x_test, y_test)
x_train, x_test, x_test_unknown = x_set
y_train, y_test, y_test_unknown = y_set

# Create training pairs
digit_idx = [np.where(y_train == i)[0] for i in range(num_classes)]
siamese_train_pairs, siamese_train_y = create_pairs(x_train, digit_idx)

# Create test pairs
digit_idx = [np.where(y_test == i)[0] for i in range(num_classes)]
siamese_test_pairs, siamese_test_y = create_pairs(x_test, digit_idx)

# Create unknown test pairs
digit_idx = [np.where(y_test_unknown == i)[0] for i in range(num_classes)]
siamese_test_unknown_pairs, siamese_test_unknown_y = create_pairs(x_test_unknown, digit_idx)

"""
3. Create CNN Architecture
"""
model = base_model(input_shape=input_shape, num_classes=num_classes)
"""
4. Siamese network
"""
# Processed left and right inputs using the model
processed_l = model(left_input)
processed_r = model(right_input)

# Merge them using distance function
# The distance function used is L2 distance (also be called Euclidean distance)
# To do this, Lambda layer is needed to wrap the distance function (writtein in lambda function) in to layer object 
distance = keras.layers.Lambda(euclidean_distance,
                  output_shape=euclidean_distance_output_shape)([processed_l, processed_r])

# Create siamese net
siamese_model = Model([processed_l, processed_r], distance)

"""
5. Train the model using pairs of data
"""
# Specify number of epochs for training
epochs = 20

siamese_model.compile(loss=constrastive_loss,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=[accuracy])

siamese_model.fit([siamese_train_pairs[:, 0], siamese_train_pairs[:, 1]], siamese_train_y,
                batch_size=128,
                epochs=epochs)

"""
6. Get train and test set
"""
train_y_pred = model.predict([siamese_train_pairs[:, 0], siamese_train_pairs[:, 1]])
train_accuracy = compute_accuracy(y_ground_truth=siamese_train_y, y_pred=train_y_pred)

test_y_pred = model.predict([siamese_test_pairs[:, 0], siamese_test_pairs[:, 1]])
test_accuracy = compute_accuracy(y_ground_truth=siamese_test_y, y_pred=test_y_pred)

test_unknown_y_pred = model.predict([siamese_test_unknown_pairs[:, 0], siamese_test_unknown_pairs[:, 1]])
test_unknown_accuracy = compute_accuracy(y_ground_truth=siamese_test_unknown_y, y_pred=test_unknown_y_pred)

print('======Siamese Network Result======')
print('Train accuracy: ' + train_accuracy)
print('Test accuracy: ' + test_accuracy)
print('Test unkown accuracy: ' + test_unknown_accuracy)
