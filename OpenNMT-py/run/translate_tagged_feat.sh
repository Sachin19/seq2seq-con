# source activate gans
# declare -a modelnames=("fren.transformer.vmf.3")
# for modelname in "${modelnames[@]}"; do
#     echo $modelname
#     for BS in  2 3 4 5 6 7 8 9 10; do
#         for pos_topk in 1; do
#             rm logs/$modelname/bleus9_${pos_topk}_$BS.proxybeam.analysis.txt
#             for i in {2000..20000..2000}; do
#             # for i in 36000;
#                 python -u translate.py\
#                     -decode_loss cosine\
#                     -gpu 0\
#                     -model logs/$modelname/model_sec_step_$i.pt\
#                     -new_vocab ../../kumarvon2018-data/fren/conmt300_tagged/data.vocab.pt\
#                     -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr\
#                     -output logs/$modelname/step_${i}_pred.postopk${pos_topk}.bs$BS.en\
#                     -batch_size 4000\
#                     -batch_type tokens\
#                     -beam_size $BS\
#                     -pos_topk 0\
#                     -multi_task\
#                     -proxy_beam\
#                     -replace_unk >> logs/$modelname/decodelogs_multi_task.out 2>&1
#                 ./evaluate.sh logs/$modelname/step_${i}_pred.postopk${pos_topk}.bs$BS.en ../../kumarvon2018-data/fren/tst201516.en en >> logs/$modelname/bleus9_${pos_topk}_$BS.proxybeam.analysis.txt
#             done
#         done
#     done
# done
# python -u translate.py -decode_loss cosine -gpu 0 -model logs/fren.transformer.pos/model_step_2000.pt -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr -output logs/fren.transformer.pos/step_2000_pred.postopk1.bs2.en -batch_size 4000 -batch_type tokens -beam_size 2 -replace_unk -multi_task -pos_topk 1 > logs/decodelogs_tagged.out 2>&1 
# python -u translate.py -decode_loss cosine -gpu 0 -model logs/fren.transformer.vmf.3/model_sec_step_10000.pt -new_vocab ../../kumarvon2018-data/fren/conmt300_tagged/data.vocab.pt -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr -output logs/fren.transformer.vmf.3/sec_step_10000_pred.postopk1.bs4.en -batch_size 4000 -batch_type tokens -beam_size 4 -pos_topk 0 -multi_task -proxy_beam -replace_unk >> logs/fren.transformer.pos/decodelogs_multi_task.out 2>&1
source activate gans
declare -a modelnames=("fren.transformer.posfeat")
rm logs/$modelname/decodelogs_multi_task.out
for modelname in "${modelnames[@]}"; do
    echo $modelname
    for BS in 2 3 4 5 6 7 8 9 10; do
        for pos_topk in 1; do
            rm logs/$modelname/bleus9_$BS.proxybeam.txt
            for i in {2000..40000..2000}; do
            # for i in 36000;
                python -u translate.py\
                    -decode_loss cosine\
                    -gpu 0\
                    -model logs/$modelname/model_step_$i.pt\
                    -src ../../kumarvon2018-data/fren/tst201516.tok.true.fr\
                    -output logs/$modelname/sec_step_${i}_pred.bs$BS.en\
                    -batch_size 4000\
                    -batch_type tokens\
                    -beam_size $BS\
                    -pos_topk 0\
                    -multi_task\
                    -proxy_beam\
                    -replace_unk >> logs/$modelname/decodelogs_multi_task.out 2>&1
                ./evaluate.sh logs/$modelname/sec_step_${i}_pred.bs$BS.en ../../kumarvon2018-data/fren/tst201516.en en >> logs/$modelname/bleus9_$BS.proxybeam.txt
            done
        done
    done
done