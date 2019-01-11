This repository contains the code for the paper

> von-Mises Fisher Loss for Training Sequence to Sequence Models with Continuouos Outputs. _Sachin Kumar_ and _Yulia Tsvetkov_

This code base has been adapted from [open-NMT](https://github.com/OpenNMT/OpenNMT-py)

# Dependencies

* __Pytorch 0.3.0__
* Python 2.7

# Quick Start

## Proprocessing the data

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
python prepare_data.py -train_src /path/to/processed/train/file.fr -train_tgt /path/to/processed/train/file.en -valid_src /path/to/processed/valid/file.fr -valid_tgt /path/to/processed/valid/file.en -save_data /path/to/save/data.pt -src_vocab_size 50000 -tgt_vocab_size 50000 -tgt_emb /path/to/target/embeddings/file -emb_dim 300 
```
* Training the models

```
ok
```


