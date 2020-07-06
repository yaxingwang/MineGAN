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
ITERS = 10000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
CHECKPOINT_STEP = 5000
SELECTING_LABEL = [9, 8] # digits are 1 and 4
NUM_PER1 = 1
NUM_PER2 =2999 
LIMIT = NUM_PER1 + NUM_PER2 
ADAPTOR_INPUT_LEN = 4
L2_LAMBDA = 0 # teahcer ouput and student output
#SELECTING_LABEL = None # digits are 1 and 4
#LIMIT = 50000
TEACHER1_MEAN = 0. 
TEACHER2_MEAN = 0.
NUMBER_TEACHER = 10 
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
MODEL_DIR_STEP2 = 'result/home1_step1_teacher2/model/WGAN_GP.model-29000'

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


    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input2'%ID_TEACHER, 128, 128, output)
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

# teacher1
#noise1 = tf.random_normal([BATCH_SIZE, 128])
#noise1, noise2 = tf.split(noise, [128, 128], 1) 
#fake_data_noise1 = Generator(BATCH_SIZE, noise = noise1 + TEACHER1_MEAN)
#gen_cost_teahcer1, disc_cost_teahcer1, l2_teahcer1, fake_data_teacher1 = teacher_model(noise = noise1, fake_data_from_student = fake_data_noise1, mode = MODE, id_= 'teacher1', dim = 64, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN)
fake_data_set = teacher_model(noise = noise, fake_data_from_student = None, mode = MODE, id_= 'teacher2', dim = 64, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN, real_data = real_data,n_classes=N_CLASSES)

gen_cost, disc_cost, pseudo_label = Super_G_D_loss(real_data, fake_data_set)

#noise2 = tf.random_normal([BATCH_SIZE, 128])
#fake_data_noise2 = Generator(BATCH_SIZE, noise = noise2 + TEACHER2_MEAN)
#_, _, _, fake_data_teacher2, _ = teacher_model(noise = noise2, fake_data_from_student = None, mode = MODE, id_= 'teacher2', dim = 64, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN, real_data = real_data)
##student_d_pseudo_real_from_teacher2 = Super_G_D_loss(real_data = fake_data_teacher2, fake_data =  fake_data_noise2)
#gen_cost_teahcer2, disc_cost_teahcer2, disc_fake_teacher2, disc_real_teacher2 = Super_G_D_loss(real_data = real_data, fake_data = fake_data_teacher2)

#disc = tf.concat([disc_fake_teacher1 , disc_fake_teacher2], axis = 1)
#pseudo_label = tf.cast(tf.argmax(disc, axis = 1), tf.int32) 

#gen_cost, disc_fake = Super_G(fake_data_teacher1, fake_data_teacher2)
#disc_cost = Super_D(real_data, disc_fake, fake_data_teacher1, fake_data_teacher2)


# concuctor classification
pred_cls = tf.cast(tf.argmax(prob_soft, dimension=1), tf.int32) 
correction_prediction = tf.equal(pred_cls, condector_label)
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
cross_entropy_sum = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =prob_feat, labels =tf.one_hot(condector_label, NUMBER_TEACHER)))


#gen_cost = gen_cost_teahcer1 + gen_cost_teahcer2  
#disc_cost = disc_cost_teahcer1 + student_d_pseudo_real_from_teacher1 + disc_cost_teahcer2 + student_d_pseudo_real_from_teacher2 
#disc_cost = disc_cost_teahcer1 + disc_cost_teahcer2  
 

# teacher

conductor_params = lib.params_with_name('%s_Adaptor_Generator'%ID_TEACHER)
#gen_params = lib.params_with_name('%s_Generator'%ID_TEACHER)
gen_params = conductor_params
disc_params = lib.params_with_name('Discriminator')

pretrained_params2 = [p for p in lib.params_with_name('teacher2_Generator')] + [p for p in lib.params_with_name('teacher_Discriminator')]  
#pretrained_params2 =  [p for p in lib.params_with_name('teacher2_Generator')] + [p for p in lib.params_with_name('teacher2_Discriminator')]

#pretrained_params = [p for p in lib.params_with_name('teacher1_Generator')] + [p for p in lib.params_with_name('teacher2_Generator')] + [p for p in lib.params_with_name('teacher1_Discriminator')] +  [p for p in lib.params_with_name('teacher2_Discriminator')]

if MODE == 'wgan':
   # gen_cost = -tf.reduce_mean(disc_fake)
   # disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

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
   # gen_cost = -tf.reduce_mean(disc_fake)
   # disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

   # alpha = tf.random_uniform(
   #     shape=[BATCH_SIZE,1], 
   #     minval=0.,
   #     maxval=1.
   # )
   # differences = fake_data - real_data
   # interpolates = real_data + (alpha*differences)
   # gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
   # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   # gradient_penalty = tf.reduce_mean((slopes-1.)**2)
   # disc_cost += LAMBDA*gradient_penalty

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

    conductor_op = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1 =0.5,
            beta2=0.9
    ).minimize(cross_entropy_sum, var_list = conductor_params)
    clip_disc_weights = None

elif MODE == 'dcgan':
   # gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
   #     disc_fake, 
   #     tf.ones_like(disc_fake)
   # ))

   # disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
   #     disc_fake, 
   #     tf.zeros_like(disc_fake)
   # ))
   # disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
   #     disc_real, 
   #     tf.ones_like(disc_real)
   # ))
   # disc_cost /= 2.

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
fixed_noise0 = tf.constant(np.random.normal(size=(100, ADAPTOR_INPUT_LEN)).astype('float32'))
# teacher1
#noise1 = tf.random_normal([BATCH_SIZE, 128])
fixed_noise,_, _  = Generator_Adaptor(100, noise = fixed_noise0) 
#fixed_noise_samples_student1 = Generator(128, noise= fixed_noise1 + TEACHER1_MEAN)
fake_data_teacher = teacher_model(noise = fixed_noise, fake_data_from_student = None, mode = MODE, id_= 'teacher2', dim = 64, batch_size = 100, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN,real_data = real_data,n_classes=N_CLASSES)

 
test_noise = tf.random_normal([1, ADAPTOR_INPUT_LEN])
test_noise,_, test_prob_soft  = Generator_Adaptor(1, noise = test_noise) 
#fixed_noise_samples_student1 = Generator(128, noise= fixed_noise1 + TEACHER1_MEAN)
test_data_teacher = teacher_model(noise = test_noise, fake_data_from_student = None, mode = MODE, id_= 'teacher2', dim = 64, batch_size = 1, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN,real_data = real_data,n_classes=N_CLASSES)

#fixed_noise = Generator_Adaptor(BATCH_SIZE, noise = fixed_noise_adaptor) 
#fixed_noise_samples_student2 = Generator(128, noise= fixed_noise1 + TEACHER2_MEAN)
#_, _, _, fixed_fake_data_teacher2,_  = teacher_model(noise = fixed_noise2, fake_data_from_student = None, mode = MODE, id_= 'teacher2', dim = 64, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM, l2_lambda = L2_LAMBDA, adaptor_input_len = ADAPTOR_INPUT_LEN, real_data = real_data)

def generate_image_gt(frame, true_dist):
    samples = session.run(real_data,feed_dict={real_data: true_dist})
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 28, 28)), 
        SAMPLES_DIR+'GT.png'
    )
def generate_image_test(frame):

    _test_prob_soft =  session.run(test_prob_soft)
    pseudo_label =_test_prob_soft.argmax() 
    _fix_img = [i for i in session.run(test_data_teacher)]
    lib.save_images.save_images(
        _fix_img[pseudo_label].reshape((1, 28, 28)), 
        TEST_DIR+'samples_{}.png'.format(frame)
    )
def generate_image(frame, true_dist):
    #_fixed_noise_samples_student1, _fixed_fake_data_teacher1, _fixed_noise_samples_student2, _fixed_fake_data_teacher2 = session.run([fixed_noise_samples_student1, fixed_fake_data_teacher1, fixed_noise_samples_student2, fixed_fake_data_teacher2])
    #_fixed_fake_data_teacher1, _fixed_fake_data_teacher2 = session.run([fake_data_teacher])
    _fix_img = [i for i in session.run(fake_data_teacher)]
    for digit, i in enumerate(_fix_img):
        if not os.path.exists(SAMPLES_DIR + '{}/'.format(digit)):
            os.makedirs(SAMPLES_DIR + '{}/'.format(digit))
        lib.save_images.save_images(
            i.reshape((100, 28, 28)), 
            SAMPLES_DIR+ '{}/'.format(digit) + 'digit_{}_samples_{}.png'.format(digit, frame)
        )
# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL)
#train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL, num_per1=NUM_PER1, num_per2 =NUM_PER2)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images,targets

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    # pretrained teacher


    # teacher1
    #ckpt_saver = tf.train.Saver(pretrained_params1)
    #ckpt_saver.restore(session, MODEL_DIR_STEP1)
    # teacher2
    ckpt_saver = tf.train.Saver(pretrained_params2)
    ckpt_saver.restore(session, MODEL_DIR_STEP2)

    ckpt_saver = tf.train.Saver(max_to_keep = 5)

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:

            if iteration > UPDATE_CONDUNCTOR:
                _, _, _cross_entropy_sum, _accuracy, _pred_cls = session.run([gen_train_op, conductor_op, cross_entropy_sum, accuracy, pred_cls], feed_dict={condector_label:np.reshape(_pseudo_label, (BATCH_SIZE, -1)), noise_data : read_noise, real_data: _data})

            else:
                _, _cross_entropy_sum, _accuracy, _pred_cls = session.run([gen_train_op, cross_entropy_sum, accuracy, pred_cls], feed_dict={condector_label:np.reshape(_pseudo_label, (BATCH_SIZE, -1)), noise_data : read_noise, real_data: _data})
            #if iteration > UPDATE_CONDUNCTOR:
            #    _, _, _cross_entropy_sum, _accuracy,  _disc_real_teacher1, _disc_fake_teacher1, _disc_real_teacher2, _disc_fake_teacher2, _pred_cls = session.run([gen_train_op, conductor_op, cross_entropy_sum, accuracy, disc_real_teacher1, disc_fake_teacher1, disc_real_teacher2, disc_fake_teacher2, pred_cls], feed_dict={condector_label:np.reshape(_pseudo_label, (BATCH_SIZE, -1)),noise_data : read_noise, real_data: _data})
            #else:
            #    _, _cross_entropy_sum, _accuracy, _disc_real_teacher1, _disc_fake_teacher1, _disc_real_teacher2, _disc_fake_teacher2, _pred_cls = session.run([gen_train_op, cross_entropy_sum, accuracy, disc_real_teacher1, disc_fake_teacher1, disc_real_teacher2, disc_fake_teacher2, pred_cls], feed_dict={condector_label:np.reshape(_pseudo_label, (BATCH_SIZE, -1)),noise_data : read_noise, real_data: _data})
            lib.plot.plot('%s/cross_entropy_sum disc cost'%SAMPLES_DIR, _cross_entropy_sum)
            lib.plot.plot('%s/accuracy'%SAMPLES_DIR, _accuracy)
            #lib.plot.plot('%s/true_accuracy'%SAMPLES_DIR, _true_accuracy)
           # lib.plot.plot('%s/disc_real_teacher1'%SAMPLES_DIR, _disc_real_teacher1.mean())
           # lib.plot.plot('%s/disc_fake_teacher1'%SAMPLES_DIR, _disc_fake_teacher1.mean())
           # lib.plot.plot('%s/disc_real_teacher2'%SAMPLES_DIR, _disc_real_teacher2.mean())
           # lib.plot.plot('%s/disc_fake_teacher2'%SAMPLES_DIR, _disc_fake_teacher2.mean())
            lib.plot.plot('%s/stastic_%d'%(SAMPLES_DIR, SELECTING_LABEL[1]), 1.0 * len(_pred_cls[_pred_cls == SELECTING_LABEL[0]]) / BATCH_SIZE)
          #  lib.plot.plot('%s/disc_real_teacher_digit3'%SAMPLES_DIR, _disc_real_teacher1[_label==3].mean())
          #  lib.plot.plot('%s/disc_real_teacher_digit8'%SAMPLES_DIR, _disc_real_teacher1[_label==8].mean())


        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data, _label = gen.next()
            read_noise = np.random.normal(size=(BATCH_SIZE, ADAPTOR_INPUT_LEN)).astype('float32')
            if iteration == 0:
                generate_image_gt(iteration, _data)

            _pseudo_label = session.run(
                pseudo_label,
                feed_dict={noise_data : read_noise})

            _ = session.run(disc_train_op,
                feed_dict={real_data: _data, condector_label:np.reshape(_pseudo_label, (BATCH_SIZE, -1)), noise_data : read_noise}
            )

            _pseudo_label = session.run(
                 pseudo_label,
                feed_dict={real_data: _data, noise_data : read_noise}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        #lib.plot.plot('%s/train disc cost'%SAMPLES_DIR, _disc_cost)
        lib.plot.plot('%s/time'%SAMPLES_DIR, time.time() - start_time)
        # visualizing the GT

        # Calculate dev loss and generate samples every 100 iters
        if (iteration < 100) or iteration % 100 == 99:
        #    dev_disc_costs = []
        #    for images,_ in dev_gen():
        #        _dev_disc_cost = session.run(
        #            disc_cost, 
        #            feed_dict={real_data: images, noise_data : np.random.normal(size=(BATCH_SIZE, ADAPTOR_INPUT_LEN)).astype('float32')}
        #        )
        #        dev_disc_costs.append(_dev_disc_cost)
            #lib.plot.plot('%s/dev disc cost'%SAMPLES_DIR, np.mean(dev_disc_costs))

            generate_image(iteration, _data)
        # Write logs every 100 iters
        if (iteration < 2000) or (iteration % 100 == 99):
            lib.plot.flush(path = LOGS_DIR)

        if iteration % CHECKPOINT_STEP == 0:
            ckpt_saver.save(session, os.path.join(MODEL_DIR, "WGAN_GP.model"), iteration)

        if iteration > 0 and iteration % 5000 == 0:
            for i_ in xrange(10000):
                 generate_image_test(i_)
                 print i_

        lib.plot.tick()
    for i in xrange(10000):
        generate_image_test(i)
        print i
