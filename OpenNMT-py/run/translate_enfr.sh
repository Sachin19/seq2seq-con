source activate gans
#declare -a modelnames=("fren_mm" "fren_mm_te") # "fren_mm_te" "fren_mm_te_ri" "fren_mm_ns" "fren_mm_ns_te" "fren_mm_ns_te_ri")
# declare -a modelnames=("fren.transformer.vmf" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4" "fren.transformer.vmf.5")
declare -a modelnames=("enfr.transformer.l2", "enfr.transformer.l2.untied") # "fren.transformer.vmf.3") #("fren.transformer.vmf.long" "fren.transformer.vmf.5" "fren.transformer.vmf.6" "fren.transformer.vmf.7") # "fren.transformer.vmf.1" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4")
BS=1
for modelname in "${modelnames[@]}"; do
    echo $modelname
    rm logs/$modelname/bleus_l2.txt
    for i in {2000..40000..2000}; do
        python -u translate.py\
            -decode_loss l2\
            -gpu 0\
            -model logs/$modelname/model_step_${i}.pt\
            -src ../../kumarvon2018-data/fren/tst201516.tok.true.en\
            -output logs/$modelname/step_${i}_pred.bs$BS.fr\
            -batch_size 4000\
            -batch_type tokens\
            -replace_unk\
            -beam_size $BS >> logs/$modelname/decodelogs_enfr.out 2>&1
        ./evaluate.sh logs/$modelname/step_${i}_pred.bs$BS.fr ../../kumarvon2018-data/fren/tst201516.fr fr >> logs/$modelname/bleus_l2.txt
    done
done
