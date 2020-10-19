DATA=../../data/parallel/it-el
SRC=$1
TGT=$2
BPE=32000
mkdir -p $DATA/onmt
python -u preprocess.py\
    -train_src $DATA/train.tok.true.clean.bpe.$BPE.$SRC\
    -train_tgt $DATA/train.tok.true.clean.bpe.$BPE.$TGT\
    -valid_src $DATA/dev.tok.true.bpe.$BPE.$SRC\
    -valid_tgt $DATA/dev.tok.true.bpe.$BPE.$TGT\
    -save_data $DATA/onmt/data\
    -src_vocab_size 32000\
    -tgt_vocab_size 32000\
    -src_seq_length 100\
    -tgt_seq_length 100 > logs/itel.out 2>&1 