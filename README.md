This repository contains the code for the paper

> [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) _Sachin Kumar_ and _Yulia Tsvetkov_

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
python scripts/prepare_data.py -train_src /path/to/processed/train/file.fr -train_tgt /path/to/processed/train/file.en \
-valid_src /path/to/processed/valid/file.fr -valid_tgt /path/to/processed/valid/file.en -save_data /path/to/save/data.pt \
-src_vocab_size 50000 -tgt_vocab_size 50000 -tgt_emb /path/to/target/embeddings/file -emb_dim 300 -normalize
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

Please follow evaluate.sh to compute BLEU score. It first detruecases and then detokenizes the output file and computes BLEU score using mult-bleu-detok.perl

## Data

Already preprocessed versions of the training, val and test data for the language pairs reported in the paper can be found [here](https://drive.google.com/file/d/1jau37sNH3axLXNndmzFAcXFoR_k4Ujhw/view?usp=sharing). Pretrained fasttext vectors: [English](https://drive.google.com/file/d/1LdzxlIx3D3MyZOKYnsX8mgOJv_qaOhfO/view?usp=sharing) and [French](https://drive.google.com/open?id=1G2sKGOmy8728pOnadMf6VjGkusOy6Tle). English vectors were trained using monolingual corpus from WMT 2016  and WMT 2014/15 for French (except common crawl).

## Pretrained Models

Pretrained models for the mentioned language pairs will soon be available as well

## Publications

If you use this code, please cite the following paper

```
@inproceedings{kumar2018vmf,
title={Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs},
author={Sachin Kumar and Yulia Tsvetkov},
booktitle={Proc. of ICLR},
year={2019},
url={https://arxiv.org/pdf/1812.04616.pdf},
}
```

## Acknowledgements

This code base has been adapted from [open-NMT](https://github.com/OpenNMT/OpenNMT-py) toolkit
scripts/compare_mt.py has been taken from [here](https://github.com/neulab/compare-mt)

