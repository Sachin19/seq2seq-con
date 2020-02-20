source activate gans
#declare -a modelnames=("fren_mm" "fren_mm_te") # "fren_mm_te" "fren_mm_te_ri" "fren_mm_ns" "fren_mm_ns_te" "fren_mm_ns_te_ri")
# declare -a modelnames=("fren.transformer.vmf" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4" "fren.transformer.vmf.5")
declare -a modelnames=("deen.transformer.vmf") # "fren.transformer.vmf.3") #("fren.transformer.vmf.long" "fren.transformer.vmf.5" "fren.transformer.vmf.6" "fren.transformer.vmf.7") # "fren.transformer.vmf.1" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4")
BS=1
for modelname in "${modelnames[@]}"; do
    echo $modelname
    for i in 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000; do
        python -u translate.py\
            -decode_loss cosine\
            -gpu 0\
            -model logs/$modelname/model_step_${i}.pt\
            -src ../../kumarvon2018-data/deen/tst201314.tok.true.de\
            -output logs/$modelname/step_${i}_pred.dev.bs$BS.en\
            -batch_size 4000\
            -batch_type tokens\
            -beam_size $BS\
            -replace_unk > logs/decodelogs_deen.out 2>&1
        ./evaluate.sh logs/$modelname/step_${i}_pred.dev.bs$BS.en ../../kumarvon2018-data/deen/tst201314.en en >> logs/$modelname/devbleus.txt
    done
done
