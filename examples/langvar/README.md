Here we describe the instructions to reproduce the results of the paper "Machine Translation into Low-resource Language Varieties". The goal is to train a translation model from a source (src) language to a target-variety (tgt) language by adapting a source to standard-variety (std) model. For example, English to Ukrainian Model by adapting a English to Russian model. 

## Step 1: Training a src-std model

# Preprocessing

* We use [fastBPE](https://github.com/glample/fastBPE) to tokenize the source and target text separately without any word tokenization as follows

```
# learning BPE codes
/path/to/fastBPE/fast learnbpe 24000 train.src > src.bpecodes
/path/to/fastBPE/fast learnbpe 24000 train.std > std.bpecodes

#applying BPE codes
/path/to/fastBPE/fast train.src.bpetok train.src src.bpecodes 
/path/to/fastBPE/fast train.std.bpetok train.std std.bpecodes 

#repeat similar steps for tokenizing the target monolingual corpus, validation set, and test set.

```
* Train std skip-gram embeddings following default hyperparameters from [Choudhary et al 2018](https://github.com/Aditi138/Embeddings)

Trained embedding tables for all language-varieties will soon be available. 

* Create preprocessed data objects for easily loading while training

```
SRCSTD_DATADIR=/path/to/save/binarized/data
mkdir -p $DATADIR
python -u preprocess.py\
    -train_src train.src.bpetok\
    -train_tgt train.std.bpetok\
    -valid_src valid.src.bpetok\
    -valid_tgt valid.std.bpetok\
    -save_data $SRCSTD_DATADIR/data\
    -tgt_emb /path/to/std/embeddings\
    -src_seq_length 175\
    -tgt_seq_length 175 
```

# Training src to std model

```
BASEDIR=/path/to/root/of/this/repository
SRCSTD_MODELDIR=/path/to/save/model
mkdir -p $SRCSTD_MODELDIR
python -u $BASEDIR/train.py\
    -data $DATADIR/data\
    -save_model $SRCSTD_MODELDIR/model\
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
    -train_steps $TRAINSTEPS\
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
    -valid_steps $VALIDSTEPS\
    -valid_batch_size 4\
    -save_checkpoint_steps $SAVESTEPS\
    -world_size 1\
    -generator_function continuous-linear\
    -loss nllvmf\
    -generator_layer_norm\
    -lambda_vmf 0.2\
    -share_decoder_embeddings\
    -gpu_ranks 0

# TRAINSTEPS, VALIDSTEPS, and SAVESTEPS are data size dependent (please see run folder for details)
# by modifying world_size (to n) and gpu_ranks (0 1 2 ... n) you can also train with n GPUs
```
This will save models at every $SAVESTEPS. We use the one with best validation BLEU score for adaptation. Use the following code to decode 

```
for i in $(eval echo {${SAVESTEPS}..${TRAINSTEPS}..${SAVESTEPS}}); do
    python -u translate.py\
        -gpu 0\
        -decode_loss cosine\
        -model $SRCSTD_MODELDIR/model_step_${i}.pt\
        -src dev.src.bpetok\
        -output $SRCSTD_MODELDIR/step_${i}.pred.std\
        -batch_size 4000\
        -batch_type tokens\
        -beam_size 1\
        -replace_unk 
    ./evaluate.sh $SRCSTD_MODELDIR/step_${i}.pred.std dev.std std
done
```

Using the same method, also train a std-src model which we will use for backtranslation (A softmax based model can also be used; both result in similar performance)

## Step 2: Adapt src-std model to src-tgt

1. Given a monolingual tgt corpus, tokenize it as follows:

```
# learning new BPE codes combining train.std and mono.tgt
cat train.std mono.tgt | /path/to/fastBPE/fast learnbpe 24000 - > std-tgt.bpecodes

#applying BPE codes
/path/to/fastBPE/fast mono.tgt.bpetok mono.tgt std-tgt.bpecodes 

#repeat similar steps for tokenizing validation set, and test set. src tokenized remains the same

```

2. Adapt the std embeddings to tgt again using the code from [Choudhary et al 2018](https://github.com/Aditi138/Embeddings) by specifying additional parameters for initializing the embeddings. 
3. Project the obtained tgt embeddings to the space of std embeddings using [MUSE](https://github.com/facebookresearch/MUSE). 
4. Using `translate.py` with the backtranslation std to src model, generate psuedo-parallel src-tgt data by translating `mono.tgt.bpetok` to src: `mono.tgt.to-src.bpetok`.
5. Finally, finetune the src-std model as follows

# Preprocessing

```
SRCTGT_DATADIR=/path/to/save/binarized/data
mkdir -p $DATADIR
python -u preprocess.py\
    -train_src mono.tgt.to-src.bpetok\
    -train_tgt mono.tgt\
    -valid_src valid.src.bpetok\
    -valid_tgt valid.tgt.bpetok\
    -save_data $SRCTGT_DATADIR/data\
    -tgt_emb /path/to/tgt/embeddings\
    -src_seq_length 175\
    -tgt_seq_length 175 
```

# Finetuning

```
BASEDIR=/path/to/root/of/this/repository
SRCTGT_MODELDIR=/path/to/save/model
mkdir -p $SRCSTD_MODELDIR
python -u train.py \
    -data $SRCTGT_DATADIR/data\
    -save_model $SRCTGT_MODELDIR/model\
    -finetune regular\
    -initialize_with /path/to/best/src-std-model\
    -num_adapters 0\
    -loss nllvmf\
    -gpu_ranks 0\
    -train_with train\
    -optim radam\
    -reset_optim all\
    -warmup_init_lr 1e-8\
    -warmup_end_lr 0.001\
    -min_lr 1e-9\
    -train_steps $TRAINSTEPS\
    -max_generator_batches 2\
    -batch_size 4000\
    -batch_type tokens\
    -normalization tokens\
    -accum_count 2\
    -adam_beta2 0.9995\
    -decay_method linear\
    -weight_decay 0.0001\
    -warmup_steps 1\
    -learning_rate 1\
    -max_grad_norm 0\
    -param_init 0\
    -param_init_glorot\
    -label_smoothing 0.1\
    -valid_steps $VALIDSTEPS\
    -valid_batch_size 4\
    -save_checkpoint_steps $SAVESTEPS\
    -world_size 1\
    -gpu_ranks 0
```

## Data

Preprocessed versions of the training, val and test data for the language pairs reported in the paper will soon be available. 

## Pretrained Models

Pretrained models for the mentioned language pairs will soon be available.
