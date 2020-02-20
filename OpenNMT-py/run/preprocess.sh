source activate gans
mkdir -p ../../kumarvon2018-data/wmt16deen/conmt300/
python -u preprocess.py\
    -train_src ../../kumarvon2018-data/wmt16deen/train.tok.clean.bpe.100000.de\
    -train_tgt ../../kumarvon2018-data/wmt16deen/train.tok.clean.filtered.en\
    -valid_src ../../kumarvon2018-data/wmt16deen/newstest2015.tok.bpe.100000.de\
    -valid_tgt ../../kumarvon2018-data/wmt16deen/newstest2015.tok.en\
    -save_data ../../kumarvon2018-data/wmt16deen/conmt300/data\
    -tgt_emb ../../../data/monolingual/en/corpus.fasttext.oldtok.txt.vec\
    -src_vocab_size 100000\
    -tgt_vocab_size 500000\
    -src_seq_length 100\
    -tgt_seq_length 100 > logs/prep_wmt16deen_oldtok.out 2>&1 