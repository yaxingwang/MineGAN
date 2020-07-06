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

from teacher import teacher_model

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
CHECKPOINT_STEP = 5000
SELECTING_LABEL = [3, 8] # digits are 1 and 4
LIMIT = 8000
L2_LAMBDA = 0
NOISE_LEN = 64 
#SELECTING_LABEL = None # digits are 1 and 4
#LIMIT = 50000
ID_STUDENT = 'student'
ID_TEACHER = 'teacher'

ADAPTOR_INPUT_LEN = 4
FOLDER_NAME = 'home1_step4/ada_input_len_%d'%ADAPTOR_INPUT_LEN
RESULT_DIR = './result/%s'%FOLDER_NAME
ADAPTOR = False

SAMPLES_DIR = RESULT_DIR + '/samples/'
LOGS_DIR = RESULT_DIR + '/log/'
MODEL_DIR = RESULT_DIR + '/model/'

# model from step3
FOLDER_NAME_STEP3 = 'home1_step3/ada_input_len_%d'%ADAPTOR_INPUT_LEN
RESULT_DIR_STEP3 = './result/%s'%FOLDER_NAME_STEP3
MODEL_STEP3 = 'WGAN_GP.model-10000'
MODEL_DIR_STEP3 = RESULT_DIR_STEP3 + '/model/%s'%MODEL_STEP3

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

def Generator_Adaptor(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 64])

    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input1'%ID_STUDENT, 64, 128, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Adaptor_Generator.BN1'%ID_STUDENT, [0], output)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input2'%ID_STUDENT, 128, 128, output)
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

    return output 
def Generator_useless(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_LEN])

    output = lib.ops.linear.Linear('%s_Generator.Input'%ID_STUDENT, NOISE_LEN, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN1'%ID_STUDENT, [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.2'%ID_STUDENT, 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN2'%ID_STUDENT, [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.3'%ID_STUDENT, 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN3'%ID_STUDENT, [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.5'%ID_STUDENT, DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])
def Generator_student(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_LEN])

    output = lib.ops.linear.Linear('%s_Generator.Input'%ID_STUDENT, NOISE_LEN, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN1'%ID_STUDENT, [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.2'%ID_STUDENT, 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN2'%ID_STUDENT, [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.3'%ID_STUDENT, 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN3'%ID_STUDENT, [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.5'%ID_STUDENT, DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('%s_Generator.Input'%ID_STUDENT, 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN1'%ID_STUDENT, [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.2'%ID_STUDENT, 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN2'%ID_STUDENT, [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.3'%ID_STUDENT, 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN3'%ID_STUDENT, [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('%s_Generator.5'%ID_STUDENT, DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.1'%ID_STUDENT,1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.2'%ID_STUDENT, DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN2'%ID_STUDENT, [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('%s_Discriminator.3'%ID_STUDENT, 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN3'%ID_STUDENT, [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('%s_Discriminator.Output'%ID_STUDENT, 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
#noise = Generator_Adaptor(BATCH_SIZE) 
noise = tf.random_normal([BATCH_SIZE, NOISE_LEN])
fake_data = Generator_student(BATCH_SIZE, noise = noise)

#disc_real = Discriminator(real_data)
#disc_fake = Discriminator(fake_data)

# teacher
noise = tf.random_normal([BATCH_SIZE, 64])
gen_cost, disc_cost, l2, _ = teacher_model(noise = noise, fake_data_from_student = fake_data, mode = MODE, id_= ID_TEACHER, dim = 64, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN, real_data = real_data)
gen_params = lib.params_with_name('%s_Generator'%ID_STUDENT)
disc_params = lib.params_with_name('%s_Discriminator'%ID_TEACHER)

#pretrained_params = [p for p in lib.params_with_name('%s_Generator'%ID_TEACHER)] + [p for p in lib.params_with_name('%s_Adaptor_Generator'%ID_TEACHER)] +  [p for p in lib.params_with_name('%s_Discriminator'%ID_TEACHER)]

pretrained_params = gen_params + disc_params 

if MODE == 'wgan':
#    gen_cost = -tf.reduce_mean(disc_fake)
#    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

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
#    gen_cost = -tf.reduce_mean(disc_fake)
#    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
#
#    alpha = tf.random_uniform(
#        shape=[BATCH_SIZE,1], 
#        minval=0.,
#        maxval=1.
#    )
#    differences = fake_data - real_data
#    interpolates = real_data + (alpha*differences)
#    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
#    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
#    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
#    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost,  var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
#    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#        disc_fake, 
#        tf.ones_like(disc_fake)
#    ))
#
#    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#        disc_fake, 
#        tf.zeros_like(disc_fake)
#    ))
#    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#        disc_real, 
#        tf.ones_like(disc_real)
#    ))
#    disc_cost /= 2.

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
fixed_noise = tf.constant(np.random.normal(size=(128, NOISE_LEN)).astype('float32'))
#fixed_noise = Generator_Adaptor(BATCH_SIZE, noise = fixed_noise_adaptor) 
fixed_noise_samples = Generator_student(128, noise=fixed_noise)


# teacher
#fixed_noise_teacher = tf.constant(np.random.normal(size=(128, 64)).astype('float32'))
#_, _, _, fixed_noise_samples_teacher = teacher_model(noise = fixed_noise_teacher, fake_data_from_student = fake_data, mode = MODE, id_= ID_TEACHER, dim = 64, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN, real_data = real_data)
#
def generate_image_gt(frame, true_dist):
    samples = session.run(real_data,feed_dict={real_data: true_dist})
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 28, 28)), 
        SAMPLES_DIR+'GT.png'
    )

def generate_teacher(frame, true_dist):
    samples = session.run(fixed_noise_samples_teacher)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        SAMPLES_DIR+'teacher.png'
    )
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        SAMPLES_DIR+'samples_{}.png'.format(frame)
    )


# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    # pretrained teacher

    ckpt_saver = tf.train.Saver(pretrained_params)
    ckpt_saver.restore(session, MODEL_DIR_STEP3)

    ckpt_saver = tf.train.Saver(max_to_keep = 100)

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            if iteration == 0:
                generate_image_gt(iteration, _data)
                #generate_teacher(iteration, _data)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('%s/train disc cost'%SAMPLES_DIR, _disc_cost)
        lib.plot.plot('%s/time'%SAMPLES_DIR, time.time() - start_time)
        # visualizing the GT

        # Calculate dev loss and generate samples every 100 iters
        if (iteration < 100) or iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('%s/dev disc cost'%SAMPLES_DIR, np.mean(dev_disc_costs))

            generate_image(iteration, _data)
        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(path = LOGS_DIR)

        if iteration % CHECKPOINT_STEP == 0:
            ckpt_saver.save(session, os.path.join(MODEL_DIR, "WGAN_GP.model"), iteration)

        lib.plot.tick()
