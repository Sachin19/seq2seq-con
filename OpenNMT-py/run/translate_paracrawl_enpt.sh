source activate gans
#declare -a modelnames=("fren_mm" "fren_mm_te") # "fren_mm_te" "fren_mm_te_ri" "fren_mm_ns" "fren_mm_ns_te" "fren_mm_ns_te_ri")
# declare -a modelnames=("fren.transformer.vmf" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4" "fren.transformer.vmf.5")
declare -a modelnames=("enpt.transformer") # "fren.transformer.vmf.3") #("fren.transformer.vmf.long" "fren.transformer.vmf.5" "fren.transformer.vmf.6" "fren.transformer.vmf.7") # "fren.transformer.vmf.1" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4")
BS=1
# for modelname in "${modelnames[@]}"; do
#     echo $modelname
#     rm logs/$modelname/decodelogs.out
#     rm logs/$modelname/bleus-wit-pteu.txt
#     for i in {10000..200000..10000}; do
#         python -u translate.py\
#             -gpu 0\
#             -model logs/$modelname/model_step_${i}.pt\
#             -src ../../../data/parallel/en-pt/wit3/test-wit-pteu.tok.true.bpe.32000.en\
#             -output logs/$modelname/step_${i}_pred.wit-pteu.bs$BS.pt\
#             -batch_size 4000\
#             -batch_type tokens\
#             -beam_size $BS\
#             -replace_unk\
#             -usenew >> logs/$modelname/decodelogs.out 2>&1
#         ./evaluate.sh logs/$modelname/step_${i}_pred.wit-pteu.bs$BS.pt ../../../data/parallel/en-pt/wit3/test-wit-pteu.pt pt >> logs/$modelname/bleus-wit-pteu.txt
#     done
# done

for modelname in "${modelnames[@]}"; do
    echo $modelname
    rm logs/$modelname/decodelogs.out
    rm logs/$modelname/bleus-wit-ptbr.txt
    for i in {10000..200000..10000}; do
        python -u translate.py\
            -gpu 0\
            -model logs/$modelname/model_step_${i}.pt\
            -src ../../../data/parallel/en-pt/wit3/test-wit-ptbr.tok.true.bpe.32000.en\
            -output logs/$modelname/step_${i}_pred.wit-ptbr.bs$BS.pt\
            -batch_size 4000\
            -batch_type tokens\
            -beam_size $BS\
            -replace_unk\
            -usenew >> logs/$modelname/decodelogs.out 2>&1
        ./evaluate.sh logs/$modelname/step_${i}_pred.wit-ptbr.bs$BS.pt ../../../data/parallel/en-pt/wit3/test-wit-ptbr.pt pt >> logs/$modelname/bleus-wit-ptbr.txt
    done
done

# 

# python -u translate.py -decode_loss cosine -gpu 0 -model logs/wmt16deen.transformer.vmf/model_step_90000.pt -src ../../kumarvon2018-data/wmt16deen/newstest2016.tok.bpe.100000.de -output logs/wmt16deen.transformer.vmf/step_90000_pred.bs1.en -batch_size 4000 -batch_type tokens -beam_size 1-usenew -phrase_table ../../../data/parallel/en-de/wmt16/de-en.filtered.align.dictstr -replace_unk
#./evaluate.sh logs/wmt16deen.transformer.vmf/step_90000_pred.bs1.en ../../../data/parallel/en-de/wmt16/newstest2016.en en >> logs/wmt16deen.transformer.vmf/bleus3.txt