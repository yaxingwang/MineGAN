#!/bin/bash
#on-manifold
python gan_mnist_knowledge_distillation_adaptor_step1.py -1 16 0 -1

python gan_mnist_knowledge_distillation_adaptor_step2.py 3,7 16 0 .1,.9
python gan_mnist_knowledge_distillation_adaptor_step2.py 3,7 16 0 .2,.8
python gan_mnist_knowledge_distillation_adaptor_step2.py 3,7 16 0 .3,.7
python gan_mnist_knowledge_distillation_adaptor_step2.py 3,7 16 0 .4,.6
python gan_mnist_knowledge_distillation_adaptor_step2.py 3,7 16 0 .5,.5

python gan_mnist_knowledge_distillation_adaptor_step3.py 3,7 16 0 .1,.9
python gan_mnist_knowledge_distillation_adaptor_step3.py 3,7 16 0 .2,.8
python gan_mnist_knowledge_distillation_adaptor_step3.py 3,7 16 0 .3,.7
python gan_mnist_knowledge_distillation_adaptor_step3.py 3,7 16 0 .4,.6
python gan_mnist_knowledge_distillation_adaptor_step3.py 3,7 16 0 .5,.5

