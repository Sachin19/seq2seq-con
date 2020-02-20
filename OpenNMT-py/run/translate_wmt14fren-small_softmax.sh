source activate gans
declare -a modelnames=("wmt14fren.small.transformer")
for modelname in "${modelnames[@]}"; do
    echo $modelname
    for BS in 1 2 3 4 5 6 7 8 9 10; do
        for pos_topk in 1; do
            rm logs/$modelname/bleus.txt
            for i in {2000..70000..2000}; do
            # for i in 36000;
                python -u translate.py\
                    -gpu 0\
                    -model logs/$modelname/model_step_$i.pt\
                    -src /projects/tir1/users/sachink/testkumarvon2018/kumarvon2018-data/wmt14fren/wmt14_en_fr/test.fr\
                    -output logs/$modelname/step_${i}_pred.bs$BS.en\
                    -batch_size 4000\
                    -batch_type tokens\
                    -beam_size $BS\
                    -replace_unk >> logs/$modelname/decodelogs.out 2>&1
                ./evaluate.sh logs/$modelname/step_${i}_pred.bs$BS.en /projects/tir1/users/sachink/testkumarvon2018/kumarvon2018-data/wmt14fren/wmt14_en_fr/test.en en >> logs/$modelname/bleus_bs$BS.txt
            done
        done
    done
done
# python -u translate.py -decode_loss cosine -gpu 0 -model logs/fren.transformer.pos/model_step_2000.pt -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr -output logs/fren.transformer.pos/step_2000_pred.postopk1.bs2.en -batch_size 4000 -batch_type tokens -beam_size 2 -replace_unk -multi_task -pos_topk 1 > logs/decodelogs_tagged.out 2>&1 

# python -u translate.py -gpu 0 -model logs/fren.transformer.softmax.pos/model_sec_step_12000.pt -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr -output logs/fren.transformer.softmax.pos/sec_step_10000_pred.postopk1.bs4.en -batch_size 4000 -batch_type tokens -beam_size 4 -pos_topk 0 -multi_task -proxy_beam -replace_unk -verbose