source activate gans
declare -a modelnames=("wmt14fren.small.transformer.vmf")
BS=1
for modelname in "${modelnames[@]}"; do
    echo $modelname
    rm logs/$modelname/decodelogs.out
    rm logs/$modelname/bleus.txt
    for i in {2000..70000..2000}; do
        python -u translate.py\
            -decode_loss cosine\
            -gpu 0\
            -model logs/$modelname/model_step_${i}.pt\
            -src /projects/tir1/users/sachink/testkumarvon2018/kumarvon2018-data/wmt14fren/wmt14_en_fr/test.fr\
            -output logs/$modelname/step_${i}_pred.bs$BS.en\
            -batch_size 4000\
            -batch_type tokens\
            -beam_size $BS\
            -usenew\
            -replace_unk >> logs/$modelname/decodelogs.out 2>&1
        ./evaluate.sh logs/$modelname/step_${i}_pred.bs$BS.en /projects/tir1/users/sachink/testkumarvon2018/kumarvon2018-data/wmt14fren/wmt14_en_fr/test.en en >> logs/$modelname/bleus.txt
    done
done



# python -u translate.py -decode_loss cosine -gpu 0 -model logs/wmt16deen.transformer.vmf/model_step_90000.pt -src ../../kumarvon2018-data/wmt16deen/newstest2016.tok.bpe.100000.de -output logs/wmt16deen.transformer.vmf/step_90000_pred.bs1.en -batch_size 4000 -batch_type tokens -beam_size 1-usenew -phrase_table ../../../data/parallel/en-de/wmt16/de-en.filtered.align.dictstr -replace_unk
#./evaluate.sh logs/wmt16deen.transformer.vmf/step_90000_pred.bs1.en ../../../data/parallel/en-de/wmt16/newstest2016.en en >> logs/wmt16deen.transformer.vmf/bleus3.txt