#!/bin/bash

## usage
#./test_portions.sh 3,7 (3 7)
#./test_portions.sh 0,1,2,4,5,6,7,8,9 3; 
#./test_portions.sh 0,1,2,3,4,5,6,8,9 7;

if [ $# -eq 0 ]
then
	BIAS_DIGITS=3,7
	#declare -a SELECTING_LABELS=(3 7)
	SELECTING_LABELS=3,7
else
	BIAS_DIGITS=$1
	declare -a SELECTING_LABELS=$2
fi


echo "step 1"
python gan_mnist_knowledge_distillation_adaptor_step1.py ${BIAS_DIGITS} 16 1 -1

echo "step 2"
python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .1,.9 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .9,.1 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .3,.7 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .7,.3 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .5,.5 ${SELECTING_LABELS}



echo "step 3"
python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .1,.9 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .9,.1 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .3,.7 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .7,.3 ${SELECTING_LABELS}
python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .5,.5 ${SELECTING_LABELS}


#individual
#for i in "${SELECTING_LABELS[@]}"; do 

	#echo "step 2"
	#python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .1,.9 $i
	#python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 .9,.1 $i

	#echo "step 3"
	#python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .1,.9 $i
	#python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 .9,.1 $i

#done



