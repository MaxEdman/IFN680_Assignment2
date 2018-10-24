'''
Scaffholding code to get you started for the 2nd assignment.

Team 10:
    
    Liying Shi N10020969
    Wai Wing Chan N9781463
    Zhiyi Wu N9589147

'''
import random
import numpy as np
from tensorflow.contrib import keras
from tensorflow.contrib.keras import backend as K
import assign2_utils
#------------------------------------------------------------------------------

def euclidean_distance(vects):

    '''
    Auxiliary function to compute the Euclidian distance between two vectors
    in a Keras layer.
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#------------------------------------------------------------------------------
def contrastive_loss(y_true, y_pred):

    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
      y_true : true label 1 for positive pair, 0 for negative pair
      y_pred : distance output of the Siamese network    

    '''
    margin = 1

    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network
    # if negative pair, y_true is 0, penalize for distance smaller than the margin

    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

#------------------------------------------------------------------------------
def compute_accuracy(predictions, labels):

    '''
    Compute classification accuracy with a fixed threshold on distances.
    @param 
      predictions : values computed by the Siamese network
      labels : 1 for positive pair, 0 otherwise

    '''
    # the formula below, compute only the true positive rate]
    # return labels[predictions.ravel() < 0.5].mean()

    n = labels.shape[0]

    acc =  (labels[predictions.ravel() < 0.5].sum() +  # count True Positive

               (1-labels[predictions.ravel() >= 0.5]).sum() ) / n  # True Negative

    return acc

#------------------------------------------------------------------------------
def create_pairs(x, digit_indices):

    '''
       Positive and negative pair creation.
       Alternates between positive and negative pairs.
       @param
         digit_indices : list of lists
            digit_indices[k] is the list of indices of occurences digit k in 
            the dataset
       @return
         P, L 
         where P is an array of pairs and L an array of labels
         L[i] ==1 if P[i] is a positive pair
         L[i] ==0 if P[i] is a negative pair
      
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):

        for i in range(n):

            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            # z1 and z2 form a positive pair

            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]

            # z1 and z2 form a negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)

#------------------------------------------------------------------------------

def create_simplistic_base_network(input_dim):

    '''
    Base network to be shared (eq. to feature extraction).
    '''
    seq = keras.models.Sequential()

    seq.add(keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_dim))

    seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))



    seq.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))

    seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))         

    seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        
    # build 4 layers convolution network
    # seq.add(keras.layers.Conv2D(128, (1, 1), activation='relu'))

    seq.add(keras.layers.Dropout(0.25))
    seq.add(keras.layers.Flatten())

    seq.add(keras.layers.Dense(576, activation='relu'))

    seq.add(keras.layers.Dropout(0.25))

    seq.add(keras.layers.Dense(128, activation='relu'))
        
    return seq
     
#------------------------------------------------------------------------------

def simplistic_solution(epochs,degree,strength):

    '''
    Train a Siamese network to predict whether two input images correspond to the 
    same digit.

    WARNING: 
        in your submission, you should use auxiliary functions to create the 
        Siamese network, to train it, and to compute its performance.

    '''
    # load dataset and expand 60,000 to 100,000 

    x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()
    a=x_train[0:40000,:,:]       

    x_train=np.concatenate((a,x_train),axis=0)

    b=y_train[0:40000]       

    y_train=np.concatenate((b,y_train),axis=0)
    img_row = x_train.shape[1]

    img_col = x_train.shape[2]

    # normalized the input image

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255 
    x_test /= 255    

    def warped_images(x_train,x_test,degree,strength):
        """
        warped dataset with degree and strength.
        
        @param: expanded x_train,x_test.
                degree: warped degree
                strength: warped strength
        @return: warped x_train, x_test
        """
        print("This time is warped data with {} degree and {} strength".format(degree, strength))
     
        for i in range(len(x_train)):

            x_train[i] = assign2_utils.random_deform(x_train[i], degree,strength)

        for i in range(len(x_test)):

            x_test[i]= assign2_utils.random_deform(x_test[i], degree,strength)
                   
        return x_train,x_test

#------------------------------------------------------------------------------

    combo_warped=False

    if combo_warped:

        x_train_easy,x_test_easy = warped_images(x_train[0:20000],x_test[0:2000],15,0.1)
        x_train_hard,x_test_hard = warped_images(x_train[20000:],x_test[2000:],45,0.3)

        x_train=np.concatenate((x_train_easy,x_train_hard),axis=0)
        x_test=np.concatenate((x_test_easy,x_test_hard),axis=0)

    else:
        x_train,x_test = warped_images(x_train,x_test,degree,strength)
    
    #reshape data    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_row, img_col)
        x_test = x_test.reshape(x_test.shape[0], 1, img_row, img_col)
        input_dim = (1, img_row, img_col)

    else:

        x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
        x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)
        input_dim = (img_row, img_col, 1)

    epochs = epochs
    
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices) 
#    print(len(te_y))
#    print(len(te_pairs))

    # network definition
    base_network = create_simplistic_base_network(input_dim)

    input_a = keras.layers.Input(shape=input_dim)
    input_b = keras.layers.Input(shape=input_dim)

    # because we re-use the same instance `base_network`,
    # the weights of the network will be shared across the two branches

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)  

    # node to compute the distance between the two vectors
    # processed_a and processed_a
    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])

    # Our model take as input a pair of images input_a and input_b
    # and output the Euclidian distance of the mapped inputs

    model = keras.models.Model([input_a, input_b], distance)
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets

    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)

    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)
    
    print('when epochs is {}'.format(epochs))
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    return tr_acc,te_acc

#------------------------------------------------------------------------------

def stage_learning():
    """
    stage learning
    training the first 20% easy warped images first,
    then training the 80% hard warped images
    test all images 
    predict the accuracy
    """
    #expand x_train,y_train into a new x_train and y_train array with 100,000 
    x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()

    a=x_train[0:40000,:,:]       
    x_train=np.concatenate((a,x_train),axis=0)
    
    b=y_train[0:40000]       
    y_train=np.concatenate((b,y_train),axis=0)
    
    img_row = x_train.shape[1]
    img_col = x_train.shape[2]

    # normalized the input image
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255 
    x_test /= 255
        
    #reshape data  
    x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
    x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)
  
    input_dim = (img_row, img_col, 1)
    
    epochs = 10
    x_train_easy=x_train
    x_train_hard=x_train
    x_test_easy=x_test
    x_test_hard=x_test

    #warped images with with easy strength and easy degree
    for i in range(100000):
        x_train_easy[i]= assign2_utils.random_deform(x_train[i], 15, 0.1)
    for i in range(100000):
        x_train_hard[i]= assign2_utils.random_deform(x_train[i],45, 0.3)
        
    #warped images with with hard strength and hard degree
    for i in range(10000):
        x_test_easy[i]= assign2_utils.random_deform(x_test[i], 15, 0.1)
    for i in range(10000):
        x_test_hard[i]= assign2_utils.random_deform(x_test[i],45, 0.3)

    # create training+test positive and negative pairs      
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs_easy, tr_y_easy = create_pairs(x_train_easy, digit_indices)
  
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs_hard, tr_y_hard = create_pairs(x_train_hard, digit_indices)
  
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    # network definition
    base_network = create_simplistic_base_network(input_dim)

    input_a = keras.layers.Input(shape=input_dim)
    input_b = keras.layers.Input(shape=input_dim)

    # because we re-use the same instance `base_network`,
    # the weights of the network will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # node to compute the distance between the two vectors
    # processed_a and processed_a
    distance= keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])

    # Our model take as input a pair of images input_a and input_b
    # and output the Euclidian distance of the mapped inputs

    #model to fit the first 20% easy warped images
    model = keras.models.Model([input_a, input_b], distance)
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs_easy[0:36096, 0], tr_pairs_easy[0:36096, 1]], tr_y_easy[0:36096],
         batch_size=128,
         epochs=epochs,
         validation_data=([te_pairs[0:3564, 0], te_pairs[0:3564, 1]], te_y[0:3564]))
    #model.save_weights("easy_warped.h5")

    #model to fit the last 80% hard warped images
    model2 = keras.models.Model([input_a, input_b], distance)
    rms = keras.optimizers.RMSprop()
    model2.compile(loss=contrastive_loss, optimizer=rms)
    
    #model2.load_weights("easy_warped.h5")
    
    model2.fit([tr_pairs_hard[36096:, 0], tr_pairs_hard[36096:, 1]], tr_y_hard[36096:],
    batch_size=128,
    epochs=epochs,
    validation_data=([te_pairs[3564:, 0], te_pairs[3564:, 1]], te_y[3564:]))
    
    #merge two tr_pairs dataset together    
    tr_pairs=np.concatenate((tr_pairs_hard[36096:],tr_pairs_easy[0:36096]),axis=0)
    tr_y = np.concatenate((tr_y_hard[36096:],tr_y_easy[0:36096]))
    
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('when epochs is {}'.format(epochs))
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%% \n\n' % (100 * te_acc))
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------        

def test(epochs,warped):
    """
    test code for experiments
    @param:
        epochs: a tuple of epochs
        warped:a tuple of (degree, stength)
    @return:
        figures to show with the same warped degree and strength,
        different epochs and their corresponding accuracy
    """
    for i in epochs:
    

        tr_list=[]
        te_list=[]

        for degree,strength in warped:
            tr_acc,te_acc=simplistic_solution(i,degree,strength)

            tr_list.append(tr_acc)
            te_list.append(te_acc)
            
        print(tr_list)
        print(te_list)            
        print(i)
       

if __name__=='__main__':

    test((20,25,30,35),((0,0),(45,0.3),(15,0.1)))
    stage_learning()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#                               CODE CEMETARY        

    

