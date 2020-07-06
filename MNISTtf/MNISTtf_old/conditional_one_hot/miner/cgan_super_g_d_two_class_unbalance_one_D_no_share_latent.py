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
import tflib.mnist
import tflib.plot
import pdb

from teacher_output_d import teacher_model

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
ID_TEACHER = 'super_student'
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 100000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
CHECKPOINT_STEP = 5000
SELECTING_LABEL = [6, -1] # digits are 1 and 4
NUM_PER1 = 1
NUM_PER2 =2999 
LIMIT = NUM_PER1 + NUM_PER2 
ADAPTOR_INPUT_LEN = 4
L2_LAMBDA = 0 # teahcer ouput and student output
#SELECTING_LABEL = None # digits are 1 and 4
#LIMIT = 50000
TEACHER1_MEAN = 0. 
TEACHER2_MEAN = 0.
NUMBER_TEACHER = 9 
UPDATE_CONDUNCTOR = 1000
N_CLASSES = NUMBER_TEACHER # number of class

FOLDER_NAME = 'home1_step2_%s'%ID_TEACHER
RESULT_DIR = './result/%s/'%FOLDER_NAME

SAMPLES_DIR = RESULT_DIR + str(SELECTING_LABEL[0])+ '/samples/'
LOGS_DIR = RESULT_DIR + str(SELECTING_LABEL[0])+ '/log/'
MODEL_DIR = RESULT_DIR + str(SELECTING_LABEL[0])+ '/model/'
TEST_DIR = RESULT_DIR +  str(SELECTING_LABEL[0])+ '/test/'

#teacher1
MODEL_DIR_STEP1 = 'result/home1_step1_teacher1/model/WGAN_GP.model-15000' 
#teacher2
MODEL_DIR_STEP2 = 'result/home1_step1_teacher2/%d/model/WGAN_GP.model-29000'%SELECTING_LABEL[0] 

MODEL_DIR_CURRENT= '/home/yaxing/NIPS2019_MNIST/conditional_one_hot/off_manifold/result/home1_step2_super_student/%d/model/WGAN_GP.model-10000'%SELECTING_LABEL[0]

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
if not os.path.exists(TEST_DIR):
  print("*** create checkpoint dir %s" % TEST_DIR)
  os.makedirs(TEST_DIR)


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

def Generator_Adaptor(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, ADAPTOR_INPUT_LEN])

    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input1'%ID_TEACHER, ADAPTOR_INPUT_LEN, 128, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Adaptor_Generator.BN1'%ID_TEACHER, [0], output)
    output = tf.nn.relu(output)

    prob_feat = lib.ops.linear.Linear('%s_Adaptor_Generator.prob'%ID_TEACHER, 128,  NUMBER_TEACHER, output)
    prob_soft = tf.nn.softmax(prob_feat)


    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input2'%ID_TEACHER, 128, 128*NUMBER_TEACHER, output)
   # output = lib.ops.deconv2d.Deconv2D('%s_Adaptor_Generator.2'%ID, 4*DIM, 2*DIM, 5, output)
   # if MODE == 'wgan':
   #     output = lib.ops.batchnorm.Batchnorm('%s_Adaptor_Generator.BN2'%ID, [0,2,3], output)
   # output = tf.nn.relu(output)

   # output = output[:,:,:7,:7]

   # output = lib.ops.deconv2d.Deconv2D('%s_Adaptor_Generator.3'%ID, 2*DIM, DIM, 5, output)
   # if MODE == 'wgan':
   #     output = lib.ops.batchnorm.Batchnorm('%s_Adaptor_Generator.BN3'%ID, [0,2,3], output)
   # output = tf.nn.relu(output)

   # output = lib.ops.deconv2d.Deconv2D('%s_Adaptor_Generator.5'%ID, DIM, 1, 5, output)
   # output = tf.nn.sigmoid(output)

    return output, prob_feat, prob_soft 
def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('%s_Generator.Input'%ID_TEACHER, 128, 4*4*4*DIM, noise)
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

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.1'%ID_TEACHER,1,DIM,5,output,stride=2)
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

    return tf.reshape(output, [BATCH_SIZE, 1])

def Super_G_D_loss(real_data, fake_data_set):
    disc_real = Discriminator(real_data)
    for i in xrange(N_CLASSES):
        fake_data = fake_data_set[i]
        disc_fake = Discriminator(fake_data)

        #gen_cost = -tf.reduce_mean(disc_fake)
        #disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        slopes = tf.reshape(slopes, [BATCH_SIZE, 1])
        #gradient_penalty = tf.reduce_mean((slopes-1.)**2)
       # disc_cost += LAMBDA*gradient_penalty

       # clip_disc_weights = None
       # gen_cost_set.append(gen_cost)
       # disc_cost_set.append(disc_cost)
        if i==0:
            disc_fake_set = disc_fake
            slopes_set =slopes 
            #gen_cost_set = gen_cost
            #disc_cost_set = disc_cost
        else:
            disc_fake_set = tf.concat([disc_fake_set, disc_fake], axis = 1)
            slopes_set = tf.concat([slopes_set, slopes], axis = 1)
            #disc_cost_set = tf.concat([disc_cost_set, disc_cost])
            #gen_cost_set = tf.concat([gen_cost_set, gen_cost])

    pseudo_label = tf.cast(tf.argmax(disc_fake_set, axis = 1), tf.int32) 
    pseudo_label_one_hot = tf.cast(tf.one_hot(pseudo_label, N_CLASSES), tf.float32)
    
    disc_fake = tf.reduce_sum(tf.multiply(disc_fake_set, pseudo_label_one_hot), reduction_indices=1) 
    slopes = tf.reduce_sum(tf.multiply(slopes_set, pseudo_label_one_hot), reduction_indices=1)
    slopes = tf.squeeze(slopes)

    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty
    clip_disc_weights = None
    

    return gen_cost, disc_cost, pseudo_label
def Super_G(fake_data_teacher1, fake_data_teacher2):
    #disc_real = Discriminator(real_data)
    Condector_label = tf.cast(condector_label, tf.float32)
    disc_fake_teacher1 = Discriminator(fake_data_teacher1)
    disc_fake_teacher2 = Discriminator(fake_data_teacher2)

    disc_fake_teacher1 = tf.multiply(disc_fake_teacher1, 1 - Condector_label)
    disc_fake_teacher2 = tf.multiply(disc_fake_teacher2, Condector_label)
    disc_fake = disc_fake_teacher1 + disc_fake_teacher2 
    

    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    elif MODE == 'wgan-gp':
        gen_cost = -tf.reduce_mean(disc_fake)

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

    return gen_cost, disc_fake
def Super_D(real_data, disc_fake, fake_data_teacher1, fake_data_teacher2):
    disc_real = Discriminator(real_data)
    #disc_fake = Discriminator(fake_data)

    Condector_label = tf.cast(condector_label, tf.float32)
    fake_data = tf.multiply(disc_fake_teacher1, 1 -Condector_label)  + tf.multiply(disc_fake_teacher2, Condector_label) 

    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

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
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

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

    return disc_cost
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM], name = 'real_data')
noise_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, ADAPTOR_INPUT_LEN], name = 'noise_data')
condector_label = tf.placeholder(tf.int32, shape = [BATCH_SIZE, 1])


noise, prob_feat, prob_soft = Generator_Adaptor(BATCH_SIZE, noise = noise_data) 


# teacher

conductor_params = lib.params_with_name('%s_Adaptor_Generator'%ID_TEACHER)
# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    ckpt_saver = tf.train.Saver(conductor_params)
    ckpt_saver.restore(session, MODEL_DIR_CURRENT)
    pdb.set_trace()
    _noise = session.run(noise)

