source activate gans
#current hyperparameters have performed the best,current-300-best model is in vmf.4
SAVEDIR='logs/enfr.transformer.l2.untied'
# export CUDA_VISIBLE_DEVICES=0
mkdir -p $SAVEDIR
python -u train.py\
    -data ../../kumarvon2018-data/fren/conmt300_enfr/data\
    -save_model $SAVEDIR/model\
    -layers 6\
    -rnn_size 512\
    -word_vec_size 512\
    -transformer_ff 1024\
    -heads 4\
    -warmup_init_lr 1e-8\
    -warmup_end_lr 0.0007\
    -min_lr 1e-9\
    -encoder_type transformer\
    -decoder_type transformer\
    -position_encoding\
    -train_steps 40000\
    -max_generator_batches 2\
    -dropout 0.1\
    -batch_size 4000\
    -batch_type tokens\
    -normalization tokens\
    -accum_count 2\
    -optim radam\
    -adam_beta2 0.9995\
    -decay_method linear\
    -weight_decay 0.00001\
    -warmup_steps 1\
    -learning_rate 1\
    -max_grad_norm 5.0\
    -param_init 0 \
    -param_init_glorot\
    -label_smoothing 0.1\
    -valid_steps 2000\
    -save_checkpoint_steps 2000\
    -world_size 1\
    -generator_function continuous-linear\
    -loss l2\
    -generator_layer_norm\
    -lambda_vmf 0.2\
    -gpu_ranks 0 > $SAVEDIR/log.out 2>&1 

    # -share_decoder_embeddings\