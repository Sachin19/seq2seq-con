source activate gans
# python -u preprocess.py\
#     -train_src ../../kumarvon2018-data/wmt16deen/train.tok.clean.bpe.100000.de\
#     -train_tgt ../../kumarvon2018-data/wmt16deen/train.tok.clean.filtered.en\
#     -valid_src ../../kumarvon2018-data/wmt16deen/newstest2015.tok.bpe.100000.de\
#     -valid_tgt ../../kumarvon2018-data/wmt16deen/newstest2015.tok.en\
#     -save_data ../../kumarvon2018-data/wmt16deen/conmt/data\
#     -tgt_emb ../../../data/monolingual/en/corpus.fasttext.txt\
#     -src_vocab_size 100000\
#     -tgt_vocab_size 500000\
#     -src_seq_length 100\
#     -tgt_seq_length 100\
#     -overwrite
mkdir -p ../../kumarvon2018-data/fren/conmt400/
python -u preprocess.py\
    -train_src ../../kumarvon2018-data/fren/train.tok.true.fr\
    -train_tgt ../../kumarvon2018-data/fren/train.tok.true.en\
    -valid_src ../../kumarvon2018-data/fren/tst201314.tok.true.fr\
    -valid_tgt ../../kumarvon2018-data/fren/tst201314.tok.true.en\
    -save_data ../../kumarvon2018-data/fren/conmt400/data\
    -tgt_emb ../../../data/monolingual/en/corpus.fasttext.400.txt\
    -src_vocab_size 50000\
    -tgt_vocab_size 50000\
    -src_seq_length 100\
    -tgt_seq_length 100\
    -overwrite
mkdir -p ../../kumarvon2018-data/fren/conmt500/
python -u preprocess.py\
    -train_src ../../kumarvon2018-data/fren/train.tok.true.fr\
    -train_tgt ../../kumarvon2018-data/fren/train.tok.true.en\
    -valid_src ../../kumarvon2018-data/fren/tst201314.tok.true.fr\
    -valid_tgt ../../kumarvon2018-data/fren/tst201314.tok.true.en\
    -save_data ../../kumarvon2018-data/fren/conmt500/data\
    -tgt_emb ../../../data/monolingual/en/corpus.fasttext.500.txt\
    -src_vocab_size 50000\
    -tgt_vocab_size 50000\
    -src_seq_length 100\
    -tgt_seq_length 100\
    -overwrite
mkdir -p ../../kumarvon2018-data/deen/conmt400/
python -u preprocess.py\
    -train_src ../../kumarvon2018-data/deen/train.tok.true.de\
    -train_tgt ../../kumarvon2018-data/deen/train.tok.true.en\
    -valid_src ../../kumarvon2018-data/deen/tst201314.tok.true.de\
    -valid_tgt ../../kumarvon2018-data/deen/tst201314.tok.true.en\
    -save_data ../../kumarvon2018-data/deen/conmt400/data\
    -tgt_emb ../../../data/monolingual/en/corpus.fasttext.400.txt\
    -src_vocab_size 50000\
    -tgt_vocab_size 50000\
    -src_seq_length 100\
    -tgt_seq_length 100\
    -overwrite
mkdir -p ../../kumarvon2018-data/deen/conmt500/
python -u preprocess.py\
    -train_src ../../kumarvon2018-data/deen/train.tok.true.de\
    -train_tgt ../../kumarvon2018-data/deen/train.tok.true.en\
    -valid_src ../../kumarvon2018-data/deen/tst201314.tok.true.de\
    -valid_tgt ../../kumarvon2018-data/deen/tst201314.tok.true.en\
    -save_data ../../kumarvon2018-data/deen/conmt500/data\
    -tgt_emb ../../../data/monolingual/en/corpus.fasttext.500.txt\
    -src_vocab_size 50000\
    -tgt_vocab_size 50000\
    -src_seq_length 100\
    -tgt_seq_length 100\
    -overwrite