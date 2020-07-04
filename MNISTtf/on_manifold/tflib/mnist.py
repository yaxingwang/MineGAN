import numpy

import os
import urllib
import gzip
#import cPickle as pickle
import pickle
import pdb
import math
import numpy as np


def rounddown(x):
    return int(math.floor(x / 100.0)) * 100

def mnist_generator(data, batch_size, n_labelled, limit=None, selecting_label = None, bias = None, portions = None):
    images, targets = data
    if bias is not None :
        #images = images[targets!=bias] 
        #targets = targets[targets!=bias] 
        manifold_targets=np.ones(len(targets),dtype=bool)*True
        for b in range(0,len(bias)):
            manifold_targets = manifold_targets*(targets != bias[b]) #bias=off_manifold
        images = images[manifold_targets]
        targets = targets[manifold_targets] 
    if selecting_label is None:
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)
    else:
        images_aux = images
        targets_aux = targets
        if portions is None :
            for label in range(0, len(selecting_label)):
                if label==0:
                    images=images_aux[targets_aux == selecting_label[label]][:int(limit/len(selecting_label))]
                    targets=targets_aux[targets_aux == selecting_label[label]][:int(limit/len(selecting_label))]
                else:
                    images=numpy.concatenate([images,images_aux[targets_aux == selecting_label[label]][:int(limit/len(selecting_label))]])
                    targets=numpy.concatenate([targets,targets_aux[targets_aux == selecting_label[label]][:int(limit/len(selecting_label))]])
        else:
            for label in range(0, len(selecting_label)):
                if label==0:
                    images=images_aux[targets_aux == selecting_label[label]][:int(limit*portions[label])]
                    targets=targets_aux[targets_aux == selecting_label[label]][:int(limit*portions[label])]
                else:
                    images=numpy.concatenate([images,images_aux[targets_aux == selecting_label[label]][:int(limit*portions[label])]])
                    targets=numpy.concatenate([targets,targets_aux[targets_aux == selecting_label[label]][:int(limit*portions[label])]])


        #images = images[targets == selecting_label[0]] 

        #targets = targets[targets == selecting_label[0]]

        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        
    #if limit is not None:

        #if len(images) > 4500:
        #    L = 4500
        #elif len(images) > 4400 and len(images) < 4500:
        #    L = 4400
        #elif len(images) > 4300 and len(images) < 4400:
        #    L = 4300
        #elif len(images) > 4200 and len(images) < 4300:
        #    L = 4200
        #else:
        #    #L = 4000
        #    #L=rounddown(len(images))
        #    L=1000

    #L=limit #fix: error if L is not divisible by batch_size
    if limit > len(images):
        L=int(math.floor(len(images)/batch_size)*batch_size)
    else:
        L=int(math.floor(limit/batch_size)*batch_size)
    print("WARNING ONLY FIRST {} MNIST DIGITS".format(L))
    images = images.astype('float32')[:L]
    targets = targets.astype('int32')[:L]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1
        
    def get_epoch():


        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)
        
        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:
            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))
        
    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None, limit= None, selecting_label = None, bias = None, portions = None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    if limit is None:
        limit=len(train_data[1])


    epoch_function_test=mnist_generator(test_data, test_batch_size, n_labelled, limit=len(test_data[1]), selecting_label = selecting_label, bias = bias, portions = portions)
    epoch_function_train=mnist_generator(train_data, batch_size, n_labelled, limit = limit, selecting_label = selecting_label, bias = bias, portions = portions)
    epoch_function_dev=mnist_generator(dev_data, test_batch_size, n_labelled, limit=len(dev_data[1]), selecting_label = selecting_label, bias = bias, portions = portions)



    return epoch_function_train,epoch_function_dev,epoch_function_test
        
        
        
        
        #mnist_generator(train_data, batch_size, n_labelled, limit=limit, selecting_label = selecting_label, bias = bias), 
        #mnist_generator(dev_data, test_batch_size, n_labelled, limit=10*test_batch_size, selecting_label = selecting_label, bias = bias), 
        #mnist_generator(test_data, test_batch_size, n_labelled, limit=10*test_batch_size, selecting_label = selecting_label, bias = bias)
    
    
