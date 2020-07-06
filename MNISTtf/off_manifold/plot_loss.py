import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import numpy as np
import tflib as lib
import tflib.plot

import pickle


if(len(sys.argv)==1):
    sys.argv=[sys.argv[0],7,16]
BIAS_DIGIT = int(sys.argv[1]) #default is 9
NOISE_LEN = int(sys.argv[2]) #default is 16
ADAPTOR_INPUT_LEN = 4

#read files
FOLDER_NAME_STEP1='home1_step1_nlen%s'%(NOISE_LEN)
FOLDER_NAME_STEP2='home1_step2_nlen%s/ada_input_len_%d'%(NOISE_LEN,ADAPTOR_INPUT_LEN)
FOLDER_NAME_STEP3='home1_step3_nlen%s/ada_input_len_%d'%(NOISE_LEN,ADAPTOR_INPUT_LEN)
FOLDER_NAME_ECCV='home1_step2_nlen%s/ada_input_len_%d_eccv'%(NOISE_LEN,ADAPTOR_INPUT_LEN)
FOLDER_NAME_SCRATCH='home1_step2_nlen%s/ada_input_len_%d_scratch'%(NOISE_LEN,ADAPTOR_INPUT_LEN)

RESULT_DIR_STEP1 = './result_20samples/%s'%FOLDER_NAME_STEP1
RESULT_DIR_STEP2 = './result_20samples/%s'%FOLDER_NAME_STEP2
RESULT_DIR_STEP3 = './result_20samples/%s'%FOLDER_NAME_STEP3
RESULT_DIR_ECCV = './result_20samples/%s'%FOLDER_NAME_ECCV
RESULT_DIR_SCRATCH = './result_20samples/%s'%FOLDER_NAME_SCRATCH

SAMPLES_DIR_STEP1 = RESULT_DIR_STEP1 + '/'+ str(BIAS_DIGIT) +  '/samples/'
SAMPLES_DIR_STEP2 = RESULT_DIR_STEP2 + '/'+ str(BIAS_DIGIT) +  '/samples/'
SAMPLES_DIR_STEP3 = RESULT_DIR_STEP3 + '/'+ str(BIAS_DIGIT) +  '/samples/'
SAMPLES_DIR_ECCV = RESULT_DIR_ECCV + '/'+ str(BIAS_DIGIT) +  '/samples/'
SAMPLES_DIR_SCRATCH = RESULT_DIR_SCRATCH + '/'+ str(BIAS_DIGIT) +  '/samples/'

#all_costs_step1=pickle.load( open( str(SAMPLES_DIR_STEP1) + "/costs_dev.pkl", "rb" ) )
all_costs_step2=pickle.load( open( str(SAMPLES_DIR_STEP2) + "/costs_dev.pkl", "rb" ) )
all_costs_step3=pickle.load( open( str(SAMPLES_DIR_STEP3) + "/costs_dev.pkl", "rb" ) )
all_costs_eccv=pickle.load( open( str(SAMPLES_DIR_ECCV) + "/costs_dev_nominer.pkl", "rb" ) )
all_costs_scratch=pickle.load( open( str(SAMPLES_DIR_SCRATCH) + "/costs_dev_nominer.pkl", "rb" ) )

#step3 uses step2, so continues loss from step2
empty_array=np.full(len(all_costs_step3),np.inf)
all_costs_step3=np.concatenate((empty_array,all_costs_step3))

#plots
plt.plot(all_costs_eccv,'-.',color="blue",label='Scratch')
plt.plot(np.negative(all_costs_scratch),'-.',color="orange",label='Transferring GAN')
plt.plot(np.negative(all_costs_step2),'-.',color="green",label='MineGAN (w/o FT)')
plt.plot(np.negative(all_costs_step3),'-.',color="red",label='MineGAN')
plt.xlabel("Iterations")
plt.ylabel("Validation Loss ($L_D$)")
#plt.ylim([np.min([all_costs_step2,all_costs_step3])-0.5,0])
#plt.ylim([-7,0])
plt.legend(prop={'size': 15})
plt.ylim(0, 10) 
plt.rc('font', size=24)
#plt.title("digit '" + str(BIAS_DIGIT)+ "'")
plt.savefig('losses_' + str(BIAS_DIGIT) + "_nlen" +  str(NOISE_LEN)  + '.pdf',bbox_inches="tight")
plt.show()

