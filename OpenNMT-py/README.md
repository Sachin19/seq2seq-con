This repository contains the code for the paper

> [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) _Sachin Kumar_ and _Yulia Tsvetkov_

# Dependencies

* __Pytorch >= 1.2__
* Python 3.6

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
see run folder
```
## Training 

```
see run folder
```

## Decoding/Translation

```
see run folder
```

## Evaluation

Please follow evaluate.sh to compute BLEU score. It first detruecases and then detokenizes the output file and computes BLEU score using mult-bleu-detok.perl

## Data

Already preprocessed versions of the training, val and test data for the language pairs reported in the paper can be found [here](https://drive.google.com/file/d/1jau37sNH3axLXNndmzFAcXFoR_k4Ujhw/view?usp=sharing). Pretrained fasttext vectors: [English](https://drive.google.com/file/d/1LdzxlIx3D3MyZOKYnsX8mgOJv_qaOhfO/view?usp=sharing) and [French](https://drive.google.com/open?id=1G2sKGOmy8728pOnadMf6VjGkusOy6Tle). English vectors were trained using monolingual corpus from WMT 2016  and WMT 2014/15 for French (except common crawl).

## Pretrained Models

Pretrained models for the mentioned language pairs will soon be available.

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

This code base has been adapted from [open-NMT](https://github.com/OpenNMT/OpenNMT-py) toolkit.

scripts/compare_mt.py has been taken from [here](https://github.com/neulab/compare-mt)

## License 

This code is freely available for non-commercial use, and may be redistributed under these conditions. Please, see the [license](https://github.com/Sachin19/seq2seq-con/blob/master/LICENSE) for further details. Interested in a commercial license? Please contact the authors

