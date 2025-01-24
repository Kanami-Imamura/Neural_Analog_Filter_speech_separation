#!/bin/bash

# Exit on error
set -e
set -o pipefail

# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
vctk2mix_wav_dir=local/sfi_vctk_2mix

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --model_type fdmgf_sigma40_32kmin --seed 39 --stage 3 --eval_mode min --vctk2mix_wav_dir ./local. --id 0,1

# General
stage=3  # Controls from which stage to start
seed=39
model_type=tdnaf_width224_32kmin
tag=${model_type}_seed${seed}  # Controls the directory name associated to the experiment

# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0

echo $tag
echo $stage

exp_name=sample

# Data
sample_rate=32000
mode=min
n_src=2  # Our model only supports n_src=2
eval_mode=min

# Training
epochs=100

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh

# Architecture
model_config=local/model/${model_type}.yml

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${n_src}sep_${sr_string}k${mode}_${uuid}
fi
expdir=exp_${exp_name}/${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le  1 ]]; then
	echo "Stage 1 : Downloading VCTK datasets and make sfi_vctk_2mix"
	# wget https://datashare.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip -O ./local/
	# unzip ./local/VCTK-Corpus.zip
	# mv ./local/VCTK-Corpus/wav48 ./local/original_vctk
	# This process takes time. The total data is approximately 24 dB.
	for sample_rate in 32000; do
		python utils/sfi_vctk_2mix.py --original_vctk_path ./local/original_vctk --metadata_path ./local/vctk_metadata --outdir_path ${vctk2mix_wav_dir}/${n_src}speakers/wav${sr_string}k --mix_type $mode --sample_rate $sample_rate --sep train
	done
	for sample_rate in `seq 8000 4000 48000`; do
		sr_string=$(($sample_rate/1000))
		python utils/sfi_vctk_2mix.py --original_vctk_path ./local/original_vctk --metadata_path ./local/vctk_metadata --outdir_path ${vctk2mix_wav_dir}/sfi_vctk_2mix/${n_src}speakers/wav${sr_string}k --mix_type $mode --sample_rate $sample_rate --sep test
	done
elif [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 32; do
	# for sr_string in 8 12 16 20 24 28 32 36 40 44 48; do
		for mode_option in min max; do
			tmp_dumpdir=data/${n_src}speakers/wav${sr_string}k/$mode_option
			echo "Generating json files in $tmp_dumpdir"
			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
			local_vctk2mix_dir=$vctk2mix_wav_dir/${n_src}speakers/wav${sr_string}k/$mode_option/
			$python_path local/preprocess_vctk2mix.py --in_dir $local_vctk2mix_dir --out_dir $tmp_dumpdir
    	done
  	done
elif [[ $stage -le 3 ]]; then
	echo "Stage 3: Training"
	mkdir -p logs
	CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--seed $seed \
		--n_src $n_src \
		--sample_rate $sample_rate \
		--epochs $epochs \
		--model_config $model_config \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
elif [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	# for eval_sample_rate in 32000
	for eval_sample_rate in `seq 8000 4000 48000`
	do
		eval_sr_string=$(($eval_sample_rate/1000))
		test_dir=data/${n_src}speakers/wav${eval_sr_string}k/${eval_mode}/tt
		CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
			--n_src 2 \
			--test_dir $test_dir \
			--use_gpu $eval_use_gpu \
			--sample_rate $eval_sample_rate \
			--mix_mode $eval_mode \
			--exp_dir $expdir | tee logs/eval_${tag}.log
		cp logs/eval_${tag}.log $expdir/eval.log
	done
fi
