# source activate gans
mkdir -p ../../kumarvon2018-data/enru/conmt300vecmap-unsup
python -u preprocess.py\
    -train_src ../../data/parallel/en-ru/preprocessed/train.tok.true.en\
    -train_tgt ../../data/parallel/en-ru/preprocessed/train.tok.true.ru\
    -valid_src ../../data/parallel/en-ru/preprocessed/newstest2015.tok.true.en\
    -valid_tgt ../../data/parallel/en-ru/preprocessed/newstest2015.tok.true.ru\
    -save_data ../../kumarvon2018-data/enru/conmt300vecmap-unsup/data\
    -tgt_emb ../../data/monolingual/ru/corpus.fasttext.unsup.ruuk\
    -src_vocab_size 80000\
    -tgt_vocab_size 80000\
    -src_seq_length 175\
    -tgt_seq_length 175 -overwrite #> logs/prep_paracrawl_enru_unsupvecmap.out 2>&1 

#  -tgt_emb ../../../data/monolingual/en/corpus.fasttext.oldtok.txt.vec\