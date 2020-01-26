MOSES_PATH="/projects/tir1/users/sachink/data/mosesdecoder"

sed -r 's/(@@ )|(@@ ?$)//g' < $1 > $1.words
python scripts/remove_duplicates.py $1.words $1.dedup
sed -r "s/(’)/'/g" < $1.dedup > $1.awords
sed -r "s/(' s)/'s/g" < $1.awords > $1.bwords
sed -r "s/( s )/ 's /g" < $1.bwords > $1.cwords
$MOSES_PATH/scripts/recaser/detruecase.perl < $1.cwords > $1.detrue
$MOSES_PATH/scripts/tokenizer/detokenizer.perl -l $3 < $1.detrue > $1.detok
./scripts/multi-bleu-detok.perl $2 < $1.detok
rm $1.words $1.dedup $1.detrue $1.detok $1.awords $1.bwords $1.cwords
