

import os, sys
sys.path.append(os.getcwd())


import numpy as np
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

def teacher_model(noise, fake_data_from_student, mode, id_, dim, batch_size, critic_iters, lambda_, iters, output_dim, l2_lambda, adaptor_input_len = 4, real_data =None, n_classes = 10, n_pixels = 28, soft=None):
    global MODE, ID, DIM, BATCH_SIZE, CRITIC_ITERS, LAMBDA, ITERS, OUTPUT_DIM, L2_LAMBDA,  ADAPTOR_INPUT_LEN, REAL_DATA, N_CLASSES, N_PIXELS
    MODE = mode 
    ID = id_
    DIM = dim
    BATCH_SIZE = batch_size
    CRITIC_ITERS = critic_iters
    LAMBDA = lambda_
    ITERS = iters
    OUTPUT_DIM = output_dim
    L2_LAMBDA = l2_lambda 
    ADAPTOR_INPUT_LEN = adaptor_input_len 
    N_CLASSES = n_classes
    N_PIXELS = n_pixels
    REAL_DATA = real_data
    #TARGET_DOMIAN = 'lsun'# Name of target domain 
    #SOURCE_DOMAIN = 'imagenet'# imagenet, places, celebA, bedroom,
    
    
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

        output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input1'%ID, ADAPTOR_INPUT_LEN, 128, noise)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Adaptor_Generator.BN1'%ID, [0], output)
        output = tf.nn.relu(output)

        output = lib.ops.linear.Linear('%s_Adaptor_Generator.Input2'%ID, 128, 128, output)
        return output
    def Generator(n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128 + N_CLASSES])

        output = lib.ops.linear.Linear('%s_Generator.Input'%ID, 128 + N_CLASSES, 4*4*4*DIM, noise)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN1'%ID, [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('%s_Generator.2'%ID, 4*DIM, 2*DIM, 5, output)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN2'%ID, [0,2,3], output)
        output = tf.nn.relu(output)

        output = output[:,:,:7,:7]

        output = lib.ops.deconv2d.Deconv2D('%s_Generator.3'%ID, 2*DIM, DIM, 5, output)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN3'%ID, [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('%s_Generator.5'%ID, DIM, 1, 5, output)
        output = tf.nn.sigmoid(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])

    def Discriminator(inputs, one_hot_tile):
        output = tf.reshape(inputs, [-1, 1, N_PIXELS, N_PIXELS])
        one_hot_tile = tf.transpose(one_hot_tile, [0, 3, 1, 2])
        output = tf.concat([output, one_hot_tile], axis = 1)

        output = lib.ops.conv2d.Conv2D('%s_Discriminator.1'%ID,1 + N_CLASSES,DIM,5,output,stride=2)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('%s_Discriminator.2'%ID, DIM, 2*DIM, 5, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN2'%ID, [0,2,3], output)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('%s_Discriminator.3'%ID, 2*DIM, 4*DIM, 5, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN3'%ID, [0,2,3], output)
        output = LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*DIM])
        output = lib.ops.linear.Linear('%s_Discriminator.Output'%ID, 4*4*4*DIM, 1, output)

        return tf.reshape(output, [-1])
    def Generator_uncondition(n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])
    
        output = lib.ops.linear.Linear('%s_Generator.Input'%ID, 128, 4*4*4*DIM, noise)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN1'%ID, [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])
    
        output = lib.ops.deconv2d.Deconv2D('%s_Generator.2'%ID, 4*DIM, 2*DIM, 5, output)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN2'%ID, [0,2,3], output)
        output = tf.nn.relu(output)
    
        output = output[:,:,:7,:7]
    
        output = lib.ops.deconv2d.Deconv2D('%s_Generator.3'%ID, 2*DIM, DIM, 5, output)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Generator.BN3'%ID, [0,2,3], output)
        output = tf.nn.relu(output)
    
        output = lib.ops.deconv2d.Deconv2D('%s_Generator.5'%ID, DIM, 1, 5, output)
        output = tf.nn.sigmoid(output)
    
        return tf.reshape(output, [-1, OUTPUT_DIM])
    
    def Discriminator_uncondition(inputs):
        output = tf.reshape(inputs, [-1, 1, 28, 28])
    
        output = lib.ops.conv2d.Conv2D('%s_Discriminator.1'%ID,1,DIM,5,output,stride=2)
        output = LeakyReLU(output)
    
        output = lib.ops.conv2d.Conv2D('%s_Discriminator.2'%ID, DIM, 2*DIM, 5, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN2'%ID, [0,2,3], output)
        output = LeakyReLU(output)
    
        output = lib.ops.conv2d.Conv2D('%s_Discriminator.3'%ID, 2*DIM, 4*DIM, 5, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('%s_Discriminator.BN3'%ID, [0,2,3], output)
        output = LeakyReLU(output)
    
        output = tf.reshape(output, [-1, 4*4*4*DIM])
        output = lib.ops.linear.Linear('%s_Discriminator.Output'%ID, 4*4*4*DIM, 1, output)
    
        return tf.reshape(output, [BATCH_SIZE, -1])

    def One_hot(real_labels, N_CLASSES):
        real_one_hot = tf.one_hot(real_labels, N_CLASSES)
        real_one_hot_tile = tf.reshape(real_one_hot, shape=[-1, 1, 1, real_one_hot.shape[-1]])
        real_one_hot_tile = tf.tile(real_one_hot_tile, multiples=[1, N_PIXELS, N_PIXELS, 1]) # expand
        return real_one_hot,  real_one_hot_tile 

    fake_data_set = []
    noise0,noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8 = tf.split(noise, num_or_size_splits=N_CLASSES, axis=1)  
    noise_set = [noise0,noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]  
    for i in xrange(9):
        noise=noise_set[i]
       # fake_real_labels = np.asarray([i for _ in xrange(BATCH_SIZE)])
       # fake_real_labels = np.asarray([3 for _ in xrange(BATCH_SIZE)])
       # fake_real_labels = tf.constant(fake_real_labels ,dtype='int32')
        #fake_real_labels = tf.random.shuffle(fake_real_labels) 
       # one_hot,  one_hot_tile =  One_hot(fake_real_labels, N_CLASSES)
        one_hot = soft
        #noise = tf.concat([tf.random_normal([BATCH_SIZE, 128]), one_hot], axis=-1)
       # if i <9 and i > 0:
       #     fake_real_labels_uselsess = np.asarray([5 for _ in xrange(BATCH_SIZE)])
       #     fake_real_labels_uselsess = tf.constant(fake_real_labels_uselsess ,dtype='int32')
       #     #fake_real_labels = tf.random.shuffle(fake_real_labels) 
       #     one_hot_useless,  _ =  One_hot(fake_real_labels_uselsess, N_CLASSES)
       #     one_hot = (1-i / 9.) * one_hot + (i / 9.) * one_hot_useless
       # elif i == 9:
       #     fake_real_labels = np.asarray([5 for _ in xrange(BATCH_SIZE)])
       #     fake_real_labels = tf.constant(fake_real_labels ,dtype='int32')
       #     #fake_real_labels = tf.random.shuffle(fake_real_labels) 
       #     one_hot,  one_hot_tile =  One_hot(fake_real_labels, N_CLASSES)
       #     #noise = tf.concat([tf.random_normal([BATCH_SIZE, 128]), one_hot], axis=-1)


        _noise = tf.concat([noise, one_hot], axis=-1)
        fake_data_set.append(Generator(BATCH_SIZE, noise = _noise))
         

    #real_data = REAL_DATA
    #
    #disc_real = Discriminator(real_data)
    #disc_fake = Discriminator(fake_data)
    #
    ##gen_params = lib.params_with_name('%s_Generator'%ID)
    ##disc_params = lib.params_with_name('%s_Discriminator'%ID)
    #
    #if MODE == 'wgan':
    #    gen_cost = -tf.reduce_mean(disc_fake)
    #    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    #    l2 = L2_LAMBDA * tf.reduce_mean(tf.square(fake_data - real_data))
    #
    #   # gen_train_op = tf.train.RMSPropOptimizer(
    #   #     learning_rate=5e-5
    #   # ).minimize(gen_cost, var_list=gen_params)
    #   # disc_train_op = tf.train.RMSPropOptimizer(
    #   #     learning_rate=5e-5
    #   # ).minimize(disc_cost, var_list=disc_params)
    #
    #   # clip_ops = []
    #   # for var in lib.params_with_name('%s_Discriminator'%ID):
    #   #     clip_bounds = [-.01, .01]
    #   #     clip_ops.append(
    #   #         tf.assign(
    #   #             var, 
    #   #             tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
    #   #         )
    #   #     )
    #   # clip_disc_weights = tf.group(*clip_ops)
    #
    #elif MODE == 'wgan-gp':
    #    gen_cost = -tf.reduce_mean(disc_fake)
    #    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    #    l2 = L2_LAMBDA * tf.reduce_mean(tf.square(fake_data - real_data))
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
    #  
    #   # gen_train_op = tf.train.AdamOptimizer(
    #   #     learning_rate=1e-4, 
    #   #     beta1=0.5,
    #   #     beta2=0.9
    #   # ).minimize(gen_cost, var_list=gen_params)
    #   # disc_train_op = tf.train.AdamOptimizer(
    #   #     learning_rate=1e-4, 
    #   #     beta1=0.5, 
    #   #     beta2=0.9
    #   # ).minimize(disc_cost, var_list=disc_params)
    #
    #   # clip_disc_weights = None
    #
    #elif MODE == 'dcgan':
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
    #    l2 = L2_LAMBDA * tf.reduce_mean(tf.square(fake_data - real_data))
    #
    #   # gen_train_op = tf.train.AdamOptimizer(
    #   #     learning_rate=2e-4, 
    #   #     beta1=0.5
    #   # ).minimize(gen_cost, var_list=gen_params)
    #   # disc_train_op = tf.train.AdamOptimizer(
    #   #     learning_rate=2e-4, 
    #   #     beta1=0.5
    #   # ).minimize(disc_cost, var_list=disc_params)
    #
    #   # clip_disc_weights = None
    
    return fake_data_set
    # For saving samples
   # fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
   # fixed_noise_samples = Generator(128, noise=fixed_noise)
   # def generate_image(frame, true_dist):
   #     samples = session.run(fixed_noise_samples)
   #     lib.save_images.save_images(
   #         samples.reshape((128, 28, 28)), 
   #         SAMPLES_DIR+'samples_{}.png'.format(frame)
   #     )
   # 
   # # Dataset iterator
   # train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
   # def inf_train_gen():
   #     while True:
   #         for images,targets in train_gen():
   #             yield images
    
    # Train loop
   # with tf.Session() as session:
   # 
   #     session.run(tf.initialize_all_variables())
   #     ckpt_saver = tf.train.Saver(max_to_keep = 100)
   # 
   #     gen = inf_train_gen()
   # 
   #     for iteration in xrange(ITERS):
   #         start_time = time.time()
   # 
   #         if iteration > 0:
   #             _ = session.run(gen_train_op)
   # 
   #         if MODE == 'dcgan':
   #             disc_iters = 1
   #         else:
   #             disc_iters = CRITIC_ITERS
   #         for i in xrange(disc_iters):
   #             _data = gen.next()
   #             _disc_cost, _ = session.run(
   #                 [disc_cost, disc_train_op],
   #                 feed_dict={real_data: _data}
   #             )
   #             if clip_disc_weights is not None:
   #                 _ = session.run(clip_disc_weights)
   # 
   #         lib.plot.plot('%s/train disc cost'%SAMPLES_DIR, _disc_cost)
   #         lib.plot.plot('%s/time'%SAMPLES_DIR, time.time() - start_time)
   # 
   #         # Calculate dev loss and generate samples every 100 iters
   #         if iteration % 100 == 99:
   #             dev_disc_costs = []
   #             for images,_ in dev_gen():
   #                 _dev_disc_cost = session.run(
   #                     disc_cost, 
   #                     feed_dict={real_data: images}
   #                 )
   #                 dev_disc_costs.append(_dev_disc_cost)
   #             lib.plot.plot('%s/dev disc cost'%SAMPLES_DIR, np.mean(dev_disc_costs))
   # 
   #             generate_image(iteration, _data)
   # 
   #         # Write logs every 100 iters
   #         if (iteration < 5) or (iteration % 100 == 99):
   #             lib.plot.flush(path = LOGS_DIR)
   # 
   #         if iteration % CHECKPOINT_STEP == 0:
   #             ckpt_saver.save(session, os.path.join(MODEL_DIR, "WGAN_GP.model"), iteration)
   # 
   #         lib.plot.tick()
