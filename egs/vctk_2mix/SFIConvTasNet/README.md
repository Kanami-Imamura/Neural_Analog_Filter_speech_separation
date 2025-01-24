# SFI Conv-TasNet

This recipe contains SFI Conv-TasNet for speech separation.
Details of this model can be found in [this paper](https://www.nowpublishers.com/article/Details/SIP-20230082).

## How to train
Quik start example
   ```
   ./run.sh --model_type fdmgf_sigma40_32kmin --seed 39 --stage 1 --eval_mode min --vctk2mix_wav_dir ./local. --id 0,1
   ```

### Datasets preparation
Stage 1 and 2 correspond to the datasets preparation.
First, download the original VCTK corpus, and then create the VCTK_2mix dataset.

### Training
Stage 3 corresponds to a training stage.
By default, the model is trained on 32 kHz data.
You can set training configurations by command-line arguments and a yaml file like `local/model`.
```
./run.sh --model_type fdmgf_sigma40_32kmin --seed 39 --stage 3 --eval_mode min --vctk2mix_wav_dir ./local/sfi_vctk_2mix --id 0,1
```

### Evaluation
Stage 4 corresponds to a evaluation stage.
By default, the model is evaluated on 8 ~ 48 kHz data.
```
./run.sh --model_type fdmgf_sigma40_32kmin --seed 39 --stage 4 --eval_mode min --vctk2mix_wav_dir ./local/sfi_vctk_2mix --id 0,1
```

## References
If you use this model, please cite the original work.
```BibTex
@article{KImamura2024ATSIP,
  author={Imamura, Kanami and Nakamura, Tomohiko and Yatabe, Kohei and Saruwatari, Hiroshi},
  journal = {APSIPA Transactions on Signal and Information Processing},
  title = {Neural Analog Filter for Sampling-Frequency-Independent Convolutional Layer},
  year=2024,
  month=nov,
  volume=13,
  number=1,
  doi={10.1561/116.20230082}
}
```

