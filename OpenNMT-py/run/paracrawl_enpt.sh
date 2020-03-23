source activate gans
#currently running=null
SAVEDIR='logs/enpt.transformer'
export CUDA_VISIBLE_DEVICES=0,1
export THC_CACHING_ALLOCATOR=0
mkdir -p $SAVEDIR
python -u train.py\
    -data ../../kumarvon2018-data/enpt/onmt/data\
    -save_model $SAVEDIR/model\
    -layers 6\
    -rnn_size 512\
    -word_vec_size 512\
    -transformer_ff 2048\
    -heads 8 \
    -encoder_type transformer\
    -decoder_type transformer\
    -position_encoding\
    -train_steps 200000 \
    -max_generator_batches 2\
    -dropout 0.1\
    -batch_size 4096\
    -batch_type tokens\
    -normalization tokens \
    -accum_count 2\
    -optim adam\
    -adam_beta2 0.998\
    -decay_method noam\
    -warmup_steps 8000\
    -learning_rate 2\
    -max_grad_norm 0\
    -param_init 0 \
    -param_init_glorot\
    -label_smoothing 0.1\
    -valid_steps 10000\
    -save_checkpoint_steps 10000\
    -world_size 2\
    -gpu_ranks 0 1 > $SAVEDIR/log.out 2>&1 
