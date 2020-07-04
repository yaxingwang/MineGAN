import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist_mask_digit
import tflib.mnist_step1
import tflib.plot
import pdb

from teacher import teacher_model

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
ID_TEACHER = 'teacher2'
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 30000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
N_PIXELS =28 
CHECKPOINT_STEP = 1000
NOISE_LEN = 128
FOLDER_NAME = 'home1_step1_teacher2'
LIMIT = None 
N_CLASSES = 10 
RESULT_DIR = './result/%s'%FOLDER_NAME
SAMPLES_DIR = RESULT_DIR  + '/samples/'
LOGS_DIR = RESULT_DIR  + '/log/'
MODEL_DIR = RESULT_DIR  + '/model/'

#TARGET_DOMIAN = 'lsun'# Name of target domain 
#SOURCE_DOMAIN = 'imagenet'# imagenet, places, celebA, bedroom,

lib.print_model_settings(locals().copy())

# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
  print("*** create sample dir %s" % SAMPLES_DIR)
  os.makedirs(SAMPLES_DIR)
if not os.path.exists(LOGS_DIR):
  print("*** create checkpoint dir %s" % LOGS_DIR)
  os.makedirs(LOGS_DIR)
if not os.path.exists(MODEL_DIR):
  print("*** create checkpoint dir %s" % MODEL_DIR)
  os.makedirs(MODEL_DIR)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_LEN + N_CLASSES])

    output = lib.ops.linear.Linear('%s_Generator.Input'%ID_TEACHER, NOISE_LEN + N_CLASSES, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN1'%ID_TEACHER, [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.2'%ID_TEACHER, 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN2'%ID_TEACHER, [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.3'%ID_TEACHER, 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN3'%ID_TEACHER, [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.5'%ID_TEACHER, DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, one_hot_tile):
    output = tf.reshape(inputs, [-1, 1, N_PIXELS, N_PIXELS])
    one_hot_tile = tf.transpose(one_hot_tile, [0, 3, 1, 2])
    output = tf.concat([output, one_hot_tile], axis = 1)

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.1'%ID_TEACHER,1 + N_CLASSES,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.2'%ID_TEACHER, DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN2'%ID_TEACHER, [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.3'%ID_TEACHER, 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN3'%ID_TEACHER, [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('%s_Discriminator.Output'%ID_TEACHER, 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def One_hot(real_labels, N_CLASSES):
    real_one_hot = tf.one_hot(real_labels, N_CLASSES)
    real_one_hot_tile = tf.reshape(real_one_hot, shape=[-1, 1, 1, real_one_hot.shape[-1]])
    real_one_hot_tile = tf.tile(real_one_hot_tile, multiples=[1, N_PIXELS, N_PIXELS, 1]) # expand
    return real_one_hot,  real_one_hot_tile 
#noise = Generator_Adaptor(BATCH_SIZE) 
#noise = tf.random_normal([n_samples, 128])
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
#fake_labels = tf.cast(tf.random_uniform([BATCH_SIZE//len(DEVICES)])*10, tf.int32)            

one_hot,  one_hot_tile =  One_hot(real_labels, N_CLASSES)
noise = tf.concat([tf.random_normal([BATCH_SIZE, 128]), one_hot], axis=-1)
fake_data = Generator(BATCH_SIZE, noise = noise)



disc_real = Discriminator(real_data, one_hot_tile)
disc_fake = Discriminator(fake_data, one_hot_tile)

# teacher
#teacher_fake_data = teacher_model(noise = noise, real_data = fake_data, mode = MODE, id_= ID, dim = DIM, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM)

gen_params = lib.params_with_name('%s_Generator'%ID_TEACHER)
disc_params = lib.params_with_name('%s_Discriminator'%ID_TEACHER)

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('%s_Discriminator'%ID_TEACHER):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates, one_hot_tile), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
samples = np.random.normal(loc=0, scale=1, size=(100, 128))
fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8, 9]*10,dtype='int32'))

fake_one_hot, _ = One_hot( fixed_labels, N_CLASSES)
noise_sample= tf.concat([tf.constant((samples).astype('float32')), fake_one_hot], axis=-1)
fixed_noise_samples = Generator(100, noise=noise_sample)
def generate_image_GT(frame, true_dist):
    samples = session.run(real_data, feed_dict={real_data: true_dist})
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 28, 28)), 
        SAMPLES_DIR+'real_{}.png'.format(frame)
    )

def generate_image(frame, true_dist, true_label):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((100, 28, 28)), 
        SAMPLES_DIR+'samples_{}.png'.format(frame)
    )

# Dataset iterator
#train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
train_gen, dev_gen, test_gen = lib.mnist_step1.load(BATCH_SIZE, BATCH_SIZE)

def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images,targets

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    ckpt_saver = tf.train.Saver(max_to_keep = 5)

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op, feed_dict={real_data: _data, real_labels: _label})

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data, _label = gen.next()
            
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data, real_labels: _label}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('%s/train disc cost'%SAMPLES_DIR, _disc_cost)
        lib.plot.plot('%s/time'%SAMPLES_DIR, time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration < 100 or iteration % 100 == 99:
       #     dev_disc_costs = []
       #     for images,_ in dev_gen():
       #         _dev_disc_cost = session.run(
       #             disc_cost, 
       #             feed_dict={real_data: images}
       #         )
       #         dev_disc_costs.append(_dev_disc_cost)
       #     lib.plot.plot('%s/dev disc cost'%SAMPLES_DIR, np.mean(dev_disc_costs))

            generate_image(iteration, _data, _label)

        if iteration == 0 :
            generate_image_GT(iteration, _data)
        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(path = LOGS_DIR)
        if iteration  == 0:
            generate_image_GT(iteration, _data)
        if iteration % CHECKPOINT_STEP == 0:
            ckpt_saver.save(session, os.path.join(MODEL_DIR, "WGAN_GP.model"), iteration)

        lib.plot.tick()
