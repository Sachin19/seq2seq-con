source activate gans
mkdir -p ../../kumarvon2018-data/enpt/conmt300vecmap
python -u preprocess.py\
    -train_src ../../../data/parallel/en-pt/train.tok.true.clean.en\
    -train_tgt ../../../data/parallel/en-pt/train.tok.true.clean.pt\
    -valid_src ../../../data/parallel/en-pt/dev.tok.true.en\
    -valid_tgt ../../../data/parallel/en-pt/dev.tok.true.pt\
    -save_data ../../kumarvon2018-data/enpt/conmt300vecmap/data\
    -src_vocab_size 60000\
    -tgt_vocab_size 60000\
    -tgt_emb ../../../data/monolingual/pt/eu/corpus.fasttext.eubr\
    -src_seq_length 100\
    -tgt_seq_length 100 > logs/prep_paracrawl_enpt_embvecmap.out 2>&1 

#  -tgt_emb ../../../data/monolingual/en/corpus.fasttext.oldtok.txt.vec\