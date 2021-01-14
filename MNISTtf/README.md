# MNIST experiments (TF 1.08)
# Installation
Install and run your docker by running (requires nvidia-docker)
```
sudo docker/build
sudo docker/run
```
# Run
Go to your on/off_manifold folder and run each training step of your experiments:
(specify the bias digit "0-9", noise vector length and gpu id)
```
python gan_mnist_knowledge_distillation_adaptor_step1.py $bias_digit $noise_len $gpu
python gan_mnist_knowledge_distillation_adaptor_step2.py $bias_digit $noise_len $gpu
python gan_mnist_knowledge_distillation_adaptor_step3.py $bias_digit $noise_len $gpu
python gan_mnist_knowledge_distillation_adaptor_step4.py $bias_digit $noise_len $gpu
```
See some examples in test_noiselen, test_portions and test_unbias scripts.

# Contact
If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: dberga@cvc.uab.es
