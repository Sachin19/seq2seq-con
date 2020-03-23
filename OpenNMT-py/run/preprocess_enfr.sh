source activate gans
mkdir -p ../../kumarvon2018-data/fren/conmt300_enfr/
python -u preprocess.py\
    -train_src ../../kumarvon2018-data/fren/train.tok.true.en\
    -train_tgt ../../kumarvon2018-data/fren/train.tok.true.fr\
    -valid_src ../../kumarvon2018-data/fren/tst201314.tok.true.en\
    -valid_tgt ../../kumarvon2018-data/fren/tst201314.tok.true.fr\
    -save_data ../../kumarvon2018-data/fren/conmt300_enfr/data\
    -tgt_emb ../../../data/monolingual/fr/corpus.fasttext.fr\
    -src_vocab_size 50000\
    -tgt_vocab_size 50000\
    -src_seq_length 100\
    -tgt_seq_length 100\
    -overwrite