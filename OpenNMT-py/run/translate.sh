source activate gans
#declare -a modelnames=("fren_mm" "fren_mm_te") # "fren_mm_te" "fren_mm_te_ri" "fren_mm_ns" "fren_mm_ns_te" "fren_mm_ns_te_ri")
# declare -a modelnames=("fren.transformer.vmf" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4" "fren.transformer.vmf.5")
declare -a modelnames=("fren.transformer.vmf") #("fren.transformer.vmf.long" "fren.transformer.vmf.5" "fren.transformer.vmf.6" "fren.transformer.vmf.7") # "fren.transformer.vmf.1" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4")
BS=1
for modelname in "${modelnames[@]}"; do
    echo $modelname
    rm logs/$modelname/bleus_negunk.txt
    for i in {2000..40000..2000}; do
        python -u translate.py\
            -decode_loss cosine\
            -gpu 0\
            -model logs/$modelname/model_step_${i}.pt\
            -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr\
            -output logs/$modelname/step_${i}_pred.bs$BS.en\
            -batch_size 4000\
            -batch_type tokens\
            -replace_unk\
            -beam_size $BS >> logs/$modelname/decodelogs.out 2>&1
        ./evaluate.sh logs/$modelname/step_${i}_pred.bs$BS.en ../../kumarvon2018-data/fren/tst201516.en en >> logs/$modelname/bleus_negunk.txt
    done
done

# python -u translate.py -decode_loss cosine -gpu 0 -model logs/fren.transformer.vmf.400/model_step_2000.pt -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr -output logs/fren.transformer.vmf.400/step_2000_pred.bs1.en -batch_size 4000 -batch_type tokens -beam_size 1 -replace_unk > logs/decodelogs.out 2>&1