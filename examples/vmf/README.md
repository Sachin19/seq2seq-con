Here we describe the instructions to reproduce the results of the paper:

| [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) _Sachin Kumar_ and _Yulia Tsvetkov_

The goal is to train a translation model from a source (src) language to a target (tgt) language.

## Proprocessing

* Tokenization and Truecasing (Using [Moses Scripts](https://github.com/moses-smt/mosesdecoder))

```
/path/to/moses/scripts/tokenizer/tokenizer.perl -l srccode -a -no-escape -threads 20 < train.src > train.tok.src
/path/to/moses/scripts/tokenizer/tokenizer.perl -l tgtcode -a -no-escape -threads 20 < train.tgt > train.tok.tgt
#repeat similar steps for tokenizing val and test sets

/path/to/moses/scripts/recaser/train-truecaser.perl --model truecaser.model.src --corpus train.tok.src
/path/to/moses/scripts/recaser/train-truecaser.perl --model truecaser.model.tgt --corpus train.tok.tgt

/path/to/moses/scripts/recaser/truecase.perl --model truecaser.model.src < train.tok.src > train.tok.true.src
/path/to/moses/scripts/recaser/truecase.perl --model truecaser.model.tgt < train.tok.tgt > train.tok.true.tgt
#repeat similar steps for truecasing val and test sets (using the same truecasing model learnt from train)
```

* Create preprocessed data objects for easily loading while training

```
mkdir -p /path/to/save/binarized/data
python -u preprocess.py\
    -train_src train.tok.true.src\
    -train_tgt train.tok.true.tgt\
    -valid_src valid.tok.true.src\
    -valid_tgt valid.tok.true.src\
    -save_data /path/to/save/binarized/data\
    -tgt_emb /path/to/tgt/embeddings\
    -src_vocab_size 50000\
    -tgt_vocab_size 50000\
    -src_seq_length 100\
    -tgt_seq_length 100
```

## Training 

```
MODELDIR='/path/to/save/model'
mkdir -p $SAVEDIR
python -u train.py\
    -data /path/to/save/binarized/data\
    -save_model $MODELDIR/model\
    -layers 6\
    -rnn_size 512\
    -word_vec_size 512\
    -transformer_ff 1024\
    -heads 4\
    -warmup_init_lr 1e-8\
    -warmup_end_lr 0.0007\
    -min_lr 1e-9\
    -encoder_type transformer\
    -decoder_type transformer\
    -position_encoding\
    -train_steps 40000\
    -max_generator_batches 2\
    -dropout 0.1\
    -batch_size 4000\
    -batch_type tokens\
    -normalization tokens\
    -accum_count 2\
    -optim radam\
    -adam_beta2 0.9995\
    -decay_method linear\
    -weight_decay 0.00001\
    -warmup_steps 1\
    -learning_rate 1\
    -max_grad_norm 5.0\
    -param_init 0 \
    -param_init_glorot\
    -label_smoothing 0.1\
    -valid_steps 5000\
    -save_checkpoint_steps 5000\
    -world_size 1\
    -generator_function continuous-linear\
    -loss nllvmf\
    -generator_layer_norm\
    -lambda_vmf 0.2\
    -share_decoder_embeddings\
    -gpu_ranks 0 
```

## Decoding/Translation

After every 5000 steps of training, a model checkpoint is saved. Select the best one based on validation BLEU scores. Use the following script to decode

```
python -u translate.py\
    -decode_loss cosine\
    -gpu 0\
    -model /path/to/model/checkpoint\
    -src test.tok.true.src\
    -output /path/to/save/output\
    -batch_size 4000\
    -batch_type tokens\
    -replace_unk\
    -beam_size 1
```

## Evaluation

```
./evaluate.sh /path/to/save/output test.tgt tgt 
```

Please follow evaluate.sh to compute BLEU score. It first detruecases and then detokenizes the output file and computes BLEU score using mult-bleu-detok.perl

## Data

Already preprocessed versions of the training, val and test data for the language pairs reported in the paper can be found [here](https://drive.google.com/file/d/1jau37sNH3axLXNndmzFAcXFoR_k4Ujhw/view?usp=sharing). Pretrained fasttext vectors: [English](https://drive.google.com/file/d/1LdzxlIx3D3MyZOKYnsX8mgOJv_qaOhfO/view?usp=sharing) and [French](https://drive.google.com/open?id=1G2sKGOmy8728pOnadMf6VjGkusOy6Tle). English vectors were trained using monolingual corpus from WMT 2016  and WMT 2014/15 for French (except common crawl).

## Pretrained Models

Pretrained models for the mentioned language pairs will soon be available.
