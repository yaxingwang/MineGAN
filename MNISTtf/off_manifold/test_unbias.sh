#!/bin/bash

#./test_unbias.sh 0,1,2,4,5,6,8,9 (3 7)
#./test_unbias.sh 0,1,2,4,5,6,8,9 3
#./test_unbias.sh 0,1,2,4,5,6,8,9 7

#./test_unbias.sh 0 (1 2 3 4 5 6 7 8 9);
#./test_unbias.sh 1 (0 2 3 4 5 6 7 8 9)
#./test_unbias.sh 2 (0 1 3 4 5 6 7 8 9);
#./test_unbias.sh 3 (0 1 2 4 5 6 7 8 9);
#./test_unbias.sh 4 (0 1 2 3 5 6 7 8 9);
#./test_unbias.sh 5 (0 1 2 3 4 6 7 8 9);
#./test_unbias.sh 6 (0 1 2 3 4 5 7 8 9);
#./test_unbias.sh 7 (0 1 2 3 4 5 6 8 9);
#./test_unbias.sh 8 (0 1 2 3 4 5 6 7 9);
#./test_unbias.sh 9 (0 1 2 3 4 5 6 7 8);

if [ $# -eq 0 ]
then
	BIAS_DIGITS="0,1,2,4,5,6,8,9"
	declare -a SELECTING_LABELS=(3 7)
else
	BIAS_DIGITS=$1
	declare -a SELECTING_LABELS=$2
fi

echo "step 1"
python gan_mnist_knowledge_distillation_adaptor_step1.py ${BIAS_DIGITS} 16 1 -1 


for i in "${SELECTING_LABELS[@]}"; do 
	#echo "python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 -1 $i"; 
	echo "step 2"
	python gan_mnist_knowledge_distillation_adaptor_step2.py ${BIAS_DIGITS} 16 1 -1 $i
	echo "step 3"
	python gan_mnist_knowledge_distillation_adaptor_step3.py ${BIAS_DIGITS} 16 1 -1 $i
done

#individual
#python gan_mnist_knowledge_distillation_adaptor_step1.py 0,1,2,4,5,6,8,9 16 1 -1 
#bias=all except 3 and 7

#python gan_mnist_knowledge_distillation_adaptor_step2.py 0,1,2,4,5,6,8,9 16 1 -1 3
#python gan_mnist_knowledge_distillation_adaptor_step2.py 0,1,2,4,5,6,8,9 16 1 -1 7

#python gan_mnist_knowledge_distillation_adaptor_step3.py 0,1,2,4,5,6,8,9 16 1 -1 3
#python gan_mnist_knowledge_distillation_adaptor_step3.py 0,1,2,4,5,6,8,9 16 1 -1 7


