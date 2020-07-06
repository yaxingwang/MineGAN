import numpy

import os
import urllib
import gzip
import cPickle as pickle
import pdb

def mnist_generator(data, batch_size, n_labelled, limit=None, selecting_label = None, bias = None):
    images, targets = data
    if bias is not None :
        images = images[targets!=bias] 
        targets = targets[targets!=bias] 
        targets_set = []
        for i in  targets:
            if i >  bias:
                targets_set.append((i - 1).copy())
            else:
                targets_set.append(i.copy())
        targets = numpy.asarray(targets_set)
    if selecting_label is None:
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)
    else:
        images_set1 = images[targets == selecting_label[0]][:limit/2] 
        images_set2 = images[targets == selecting_label[1]][:limit/2]
        images = numpy.concatenate([images_set1, images_set2])

        targets_set1 = targets[targets == selecting_label[0]][:limit/2]
        targets_set2 = targets[targets == selecting_label[1]][:limit/2]
        targets = numpy.concatenate([targets_set1, targets_set2])

        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

    if limit is not None:
        if len(images) > 45000:
            L = 45000
        elif len(images) > 44000 and len(images) < 45000:
            L = 44000
        elif len(images) > 43000 and len(images) < 44000:
            L = 43000
        elif len(images) > 42000 and len(images) < 43000:
            L = 42000
        else:
            L = 40000
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(L)
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

def load(batch_size, test_batch_size, n_labelled=None, limit= 50000, selecting_label = None, bias = None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size, n_labelled, limit=limit, selecting_label = selecting_label, bias = bias), 
        mnist_generator(dev_data, test_batch_size, n_labelled, limit=10*test_batch_size, selecting_label = selecting_label, bias = bias), 
        mnist_generator(test_data, test_batch_size, n_labelled, limit=10*test_batch_size, selecting_label = selecting_label, bias = bias)
    )
