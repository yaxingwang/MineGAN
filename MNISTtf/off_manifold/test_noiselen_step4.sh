#/bin/bash
bias_nums="9 8 7 6 5 4 3 2 1 0"
noise_latent_sizes="8 16" #"128 64 32 16 8"

for i in $bias_nums; do list_biases="$list_biases $i"; done
for i in $noise_latent_sizes; do list_noise_lens="$list_noise_lens $i"; done

for bias_digit in $list_biases; do
	for noise_len in $list_noise_lens; do
		if [ ! -d "result/home1_step4_nlen${noise_len}/ada_input_len_4/$bias_digit" ]; then
			python gan_mnist_knowledge_distillation_adaptor_step4.py $bias_digit $noise_len;
			#echo $bias_digit
			#echo $noise_len
		fi
	done
done


