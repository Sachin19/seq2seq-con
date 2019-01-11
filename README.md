This repository contains the code for the paper

> von-Mises Fisher Loss for Training Sequence to Sequence Models with Continuouos Outputs. _Sachin Kumar_ and _Yulia Tsvetkov_

# Dependencies

* __Pytorch 0.3.0__
* Python 2.7

# Quick Start

## Proprocessing

* Tokenization and Truecasing (Using [Moses Scripts](https://github.com/moses-smt/mosesdecoder))

```
/path/to/moses/scripts/tokenizer/tokenizer.perl -l en -a -no-escape -threads 20 < train.en > train.tok.en
/path/to/moses/scripts/tokenizer/tokenizer.perl -l fr -a -no-escape -threads 20 < train.fr > train.tok.fr
#repeat similar steps for tokenizing val and test sets

/path/to/moses/scripts/recaser/train-truecaser.perl --model truecaser.model.en --corpus train.tok.en
/path/to/moses/scripts/recaser/train-truecaser.perl --model truecaser.model.fr --corpus train.tok.fr

/path/to/moses/scripts/recaser/truecase.perl --model truecaser.model.en < train.tok.en > train.tok.true.en
/path/to/moses/scripts/recaser/truecase.perl --model truecaser.model.fr < train.tok.fr > train.tok.true.fr
#repeat similar steps for truecasing val and test sets (using the same truecasing model learnt from train)
```

* Create preprocessed data objects for easily loading while training

```
python prepare_data.py -train_src /path/to/processed/train/file.fr -train_tgt /path/to/processed/train/file.en \
-valid_src /path/to/processed/valid/file.fr -valid_tgt /path/to/processed/valid/file.en -save_data /path/to/save/data.pt \
-src_vocab_size 50000 -tgt_vocab_size 50000 -tgt_emb /path/to/target/embeddings/file -emb_dim 300 
```
## Training 

```
python train.py -gpus 0 -data /path/to/save/data.pt -layers 2 -rnn_size 1024 -word_vec_size 512 -output_emb_size 300 -brnn -loss nllvmf -epochs 15 -optim adam -dropout 0.0 -learning_rate 0.0005 -log_interval 100 -save_model /path/to/save/model -batch_size 64 -tie_emb
```

## Decoding/Translation

```
python translate.py -loss nllvmf -gpu 0 -model /path/to/save/model -src /path/to/test/file.fr -tgt /path/to/test/file.en -replace_unk -output /path/to/write/predictions -batch_size 512 -beam_size 1 -lookup_dict /path/to/lookup/dict
```

## Evaluation
evaluate.sh can be used to computer BLEU score. It first detruecases and then detokenizes the output file and computes BLEU score using mult-bleu-detok.perl
```
./evaluate.sh /path/to/predictions /path/to/target/file language_code
#language code can be something like en,fr
```

## Data

Already preprocessed versions of the training, val and test data can be found here.

## Pretrained Models

Pretrained models for the 4 pairs will soon be available as well

## Publications

If you use this code, please cite the following paper

```
@inproceedings{kumar2018von,
title={Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs},
author={Sachin Kumar and Yulia Tsvetkov},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=rJlDnoA5Y7},
}
```

## Acknowledgements

This code base has been adapted from [open-NMT](https://github.com/OpenNMT/OpenNMT-py) toolkit

