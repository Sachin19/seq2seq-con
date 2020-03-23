source activate gans
mkdir -p ../../kumarvon2018-data/enru/onmt
python -u preprocess.py\
    -train_src ../../../data/parallel/en-ru/preprocessed/train.tok.true.clean.bpe.32000.en\
    -train_tgt ../../../data/parallel/en-ru/preprocessed/train.tok.true.clean.bpe.32000.ru\
    -valid_src ../../../data/parallel/en-ru/preprocessed/newstest2015.tok.true.bpe.32000.en\
    -valid_tgt ../../../data/parallel/en-ru/preprocessed/newstest2015.tok.true.bpe.32000.ru\
    -save_data ../../kumarvon2018-data/enru/onmt/data\
    -src_vocab_size 33000\
    -tgt_vocab_size 33000\
    -src_seq_length 175\
    -tgt_seq_length 175 -overwrite > logs/prep_paracrawl_enru.out 2>&1 

#  -tgt_emb ../../../data/monolingual/en/corpus.fasttext.oldtok.txt.vec\