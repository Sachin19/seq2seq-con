source activate gans
#currently running=null
SAVEDIR='logs/wmt16deen.transformer.vmf.4'
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p $SAVEDIR
python -u -W ignore train.py\
    -data ../../kumarvon2018-data/wmt16deen/conmt/data\
    -save_model $SAVEDIR/model\
    -layers 6\
    -rnn_size 512\
    -word_vec_size 512\
    -transformer_ff 2048\
    -heads 8\
    -warmup_init_lr 1e-8\
    -warmup_end_lr 0.0007\
    -min_lr 1e-9\
    -encoder_type transformer\
    -decoder_type transformer\
    -position_encoding\
    -train_steps 200000\
    -max_generator_batches 2\
    -dropout 0.1\
    -batch_size 4096\
    -batch_type tokens\
    -normalization tokens\
    -accum_count 2\
    -optim radam\
    -adam_beta2 0.9995\
    -decay_method linear\
    -weight_decay 0.00001\
    -warmup_steps 1\
    -learning_rate 1\
    -max_grad_norm 10.0\
    -param_init 0 \
    -param_init_glorot\
    -label_smoothing 0.1\
    -valid_steps 2000\
    -save_checkpoint_steps 10000\
    -world_size 2\
    -generator_function continuous-linear\
    -loss nllvmf\
    -share_decoder_embeddings\
    -generator_layer_norm\
    -lambda_vmf 0.2\
    -gpu_ranks 0 1 > $SAVEDIR/log.out 2>&1 
