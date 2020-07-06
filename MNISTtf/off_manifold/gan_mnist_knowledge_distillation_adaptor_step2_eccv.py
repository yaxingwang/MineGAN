import os, sys
import numpy as np
sys.path.append(os.getcwd())
if(len(sys.argv)==1):
    sys.argv=[sys.argv[0],'0,1,2,4,5,6,8,9','16','0','-1','3'] #sys.argv=[sys.argv[0],'3,7','16','0','.3,.7','3,7']
if(len(sys.argv)>2):
    BIAS_DIGIT = np.array(sys.argv[1].split(',')).astype(int) #default is 9
    NOISE_LEN = int(sys.argv[2]) #default is 128
if(len(sys.argv)>3):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[3])
if(len(sys.argv)>4):
    PORTIONS=np.array(sys.argv[4].split(',')).astype(float)
if(len(sys.argv)>5):
    SELECTING_LABEL=np.array(sys.argv[5].split(',')).astype(int)
else:
    SELECTING_LABEL = BIAS_DIGIT# digits are 1 and 4


import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
import pickle

#note: load pretrained model from step 1, miner (Genrator_Adaptor) adapts source to target BIAS_DIGIT

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
ID_TEACHER = 'teacher'
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS_PREV = 10000
ITERS = 20000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
CHECKPOINT_STEP = 1000
N_SAMPLES=20
LIMIT = 1000
ADAPTOR_INPUT_LEN = 4
#SELECTING_LABEL = None # digits are 1 and 4
#LIMIT = 50000

FOLDER_NAME = 'home1_step2_nlen%s/ada_input_len_%d_eccv'%(NOISE_LEN,ADAPTOR_INPUT_LEN)
RESULT_DIR = './result/%s'%FOLDER_NAME
ADAPTOR = True

SELECTING_LABEL=np.sort(SELECTING_LABEL)
BIAS_DIGIT=np.sort(BIAS_DIGIT)

if not np.array_equal(PORTIONS,[-1]):
    LABEL_DIR=''.join(np.array(SELECTING_LABEL).astype(str)) + '_'+ ''.join([portion + "%" for portion in np.array(np.array(PORTIONS*100).astype(int)).astype(str)])
else:
    LABEL_DIR=''.join(np.array(SELECTING_LABEL).astype(str))

if np.sum(SELECTING_LABEL == BIAS_DIGIT)!=len(SELECTING_LABEL): #SELECTING_LABEL is not BIAS_DIGIT:
    LABEL_DIR=''.join(np.array(BIAS_DIGIT).astype(str)) + '_' + LABEL_DIR

if ADAPTOR: 
    SAMPLES_DIR = RESULT_DIR+ '/'+ LABEL_DIR + '/samples/'
    LOGS_DIR = RESULT_DIR + '/'+ LABEL_DIR + '/log/'
    MODEL_DIR = RESULT_DIR + '/'+ LABEL_DIR + '/model/'
    TEST_DIR = RESULT_DIR + '/'+ LABEL_DIR + '/test/'

    # model from step1
    FOLDER_NAME_STEP1 = 'home1_step1_nlen%s'%NOISE_LEN
    RESULT_DIR_STEP1 = './result/%s'%FOLDER_NAME_STEP1
    MODEL_STEP1 = 'WGAN_GP.model-%s'%str(ITERS_PREV-CHECKPOINT_STEP)
    LABEL_DIR_STEP1=''.join(np.array(BIAS_DIGIT).astype(str))
    MODEL_DIR_STEP1 = RESULT_DIR_STEP1 + '/'+ LABEL_DIR + '/model/%s'%MODEL_STEP1
else:
    SAMPLES_DIR = RESULT_DIR + '/'+ LABEL_DIR + '/samples_teacher/'
    LOGS_DIR = RESULT_DIR + '/'+ LABEL_DIR + '/log_teacher/'
    MODEL_DIR = RESULT_DIR + '/'+ LABEL_DIR + '/model_teacher/'

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

    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input1'%ID_TEACHER, ADAPTOR_INPUT_LEN, NOISE_LEN, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('%s_Adaptor_Generator.BN1'%ID_TEACHER, [0], output)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input2'%ID_TEACHER, NOISE_LEN, NOISE_LEN, output)
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
def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_LEN])

    output = lib.ops.linear.Linear('%s_Generator.Input'%ID_TEACHER, NOISE_LEN, 4*4*4*DIM, noise)
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

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
#noise = Generator_Adaptor(BATCH_SIZE) 
noise=tf.random_normal([BATCH_SIZE, NOISE_LEN])
fake_data = Generator(BATCH_SIZE, noise = noise)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# teacher
#teacher_fake_data = teacher_model(noise = noise, real_data = fake_data, mode = MODE, id_= ID, dim = DIM, batch_size = BATCH_SIZE, critic_iters = CRITIC_ITERS, lambda_= LAMBDA, iters = ITERS, output_dim = OUTPUT_DIM)


gen_params = lib.params_with_name('%s_Generator'%ID_TEACHER)
disc_params = lib.params_with_name('%s_Discriminator'%ID_TEACHER)

pretrained_params = [p for p in lib.params_with_name('%s_Generator'%ID_TEACHER)] + [p for p in lib.params_with_name('%s_Discriminator'%ID_TEACHER)]

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
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
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
fixed_noise = tf.constant(np.random.normal(size=(N_SAMPLES, NOISE_LEN)).astype('float32'))
#fixed_noise = Generator_Adaptor(BATCH_SIZE, noise = fixed_noise_adaptor) 
fixed_noise_samples = Generator(N_SAMPLES, noise=fixed_noise)

test_noise = tf.random_normal([1, NOISE_LEN])
#test_noise = Generator_Adaptor(1, noise = test_noise) 
test_samples = Generator(1, noise=test_noise)

def generate_image_gt(frame, true_dist):
    samples = session.run(real_data,feed_dict={real_data: true_dist})
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 28, 28)), 
        SAMPLES_DIR+'GT.png'
    )
def generate_image_test(frame):
    samples = session.run(test_samples)
    lib.save_images.save_images(
        samples.reshape((1, 28, 28)), 
        TEST_DIR+'samples_{}.png'.format(frame)
    )
def generate_image_test_iter(frame,it):
    samples = session.run(test_samples)
    if not os.path.exists(TEST_DIR+"/"+str(it)):
        os.makedirs(TEST_DIR+"/"+str(it))
    lib.save_images.save_images(
        samples.reshape((1, 28, 28)), 
        TEST_DIR+"/"+str(it)+"/"+'samples_{}.png'.format(frame)
    )
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((N_SAMPLES, 28, 28)), 
        SAMPLES_DIR+'samples_{}.png'.format(frame)
    )

# Dataset iterator
if not np.array_equal(PORTIONS,[-1]):
    train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL, portions = PORTIONS)
else:
    train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL)
#train_gen= lib.mnist.load_train(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL)
#dev_gen= lib.mnist.load_dev(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL)
#test_gen= lib.mnist.load_test(BATCH_SIZE, BATCH_SIZE, limit = LIMIT, selecting_label = SELECTING_LABEL)

def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

def inf_test_gen():
    while True:
        for images,targets in test_gen():
            yield images

def inf_dev_gen():
    while True:
        for images,targets in dev_gen():
            yield images


# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    # pretrained teacher
    
    if ADAPTOR: 
        ckpt_saver = tf.train.Saver(pretrained_params)
        ckpt_saver.restore(session, MODEL_DIR_STEP1)
        #print("---SELECTED PARAMS TO LOAD---")
        #print(pretrained_params)
        #print("---LOADING PARAMS---")
        #print(tf.train.list_variables(MODEL_DIR_STEP2))
        

    ckpt_saver = tf.train.Saver(max_to_keep = 5)

    gen = inf_train_gen()
    dgen = inf_dev_gen()
    #tgen = inf_test_gen()
    all_costs=[]
    all_costs_dev=[]
    
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
            _data_val = dgen.next()
            
            if iteration == 0:
                generate_image_gt(iteration, _data)
            #FID_train= FID(real_data,_data)
            #FID_val= FID(real_data,_data_val)
            
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            _dev_disc_cost = session.run(
                disc_cost,
                feed_dict={real_data: _data_val}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('%s/train disc cost'%SAMPLES_DIR, _disc_cost)
        lib.plot.plot('%s/val disc cost'%SAMPLES_DIR, _dev_disc_cost)
        lib.plot.plot('%s/time'%SAMPLES_DIR, time.time() - start_time)
        all_costs.append(_disc_cost)
        all_costs_dev.append(_dev_disc_cost)
        # visualizing the GT

        # Calculate dev loss and generate samples every 100 iters
        if (iteration < 100) or iteration % 100 == 99:
#            dev_disc_costs = []
#            for images,_ in dev_gen():
#                _dev_disc_cost = session.run(
#                    disc_cost, 
#                    feed_dict={real_data: images}
#                )
#                dev_disc_costs.append(_dev_disc_cost)
#            lib.plot.plot('%s/dev disc cost'%SAMPLES_DIR, np.mean(dev_disc_costs))

            generate_image(iteration, _data)
            #generate_image(iteration, _data_val)
            
        # Write logs every 100 iters
        if iteration == 0:
            generate_image_gt(iteration, _data)
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(path = LOGS_DIR)

        if iteration % CHECKPOINT_STEP == 0:
            ckpt_saver.save(session, os.path.join(MODEL_DIR, "WGAN_GP.model"), iteration)

        lib.plot.tick()
        #print("generating one by one generation test images")
        #for i in xrange(1000):
        #    generate_image_test_iter(i,iteration)
            #print i
    pickle.dump( all_costs, open( "%s/costs_nominer.pkl"%SAMPLES_DIR, "wb" ) )
    pickle.dump( all_costs_dev, open( "%s/costs_dev_nominer.pkl"%SAMPLES_DIR, "wb" ) )
