source activate gans
#declare -a modelnames=("fren_mm" "fren_mm_te") # "fren_mm_te" "fren_mm_te_ri" "fren_mm_ns" "fren_mm_ns_te" "fren_mm_ns_te_ri")
# declare -a modelnames=("fren.transformer.vmf" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4" "fren.transformer.vmf.5")
declare -a modelnames=("deen.transformer.l2") # "fren.transformer.vmf.3") #("fren.transformer.vmf.long" "fren.transformer.vmf.5" "fren.transformer.vmf.6" "fren.transformer.vmf.7") # "fren.transformer.vmf.1" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4")
BS=1
rm logs/$modelname/decodelogs_l2.out
for modelname in "${modelnames[@]}"; do
    echo $modelname
    for i in {2000..40000..2000}; do
        python -u translate.py\
            -decode_loss l2\
            -gpu 0\
            -model logs/$modelname/model_step_${i}.pt\
            -src ../../kumarvon2018-data/deen/tst201516.tok.true.de\
            -output logs/$modelname/step_${i}_pred.bs$BS.en\
            -batch_size 4000\
            -batch_type tokens\
            -beam_size $BS\
            -replace_unk >> logs/$modelname/decodelogs.out 2>&1
        ./evaluate.sh logs/$modelname/step_${i}_pred.bs$BS.en ../../kumarvon2018-data/deen/tst201516.en en >> logs/$modelname/bleus_l2.txt
    done
done
