export TOKENIZERS_PARALLELISM=false

current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="run_picapic_t2v512_dpo"                            # experiment name 
CONFIG='configs/train/image_dpo_ft_video/config.yaml' # experiment config 
LOGDIR="./results/imagedpo/"                                     # experiment saving directory

# run
python scripts/train.py \
-t --devices '7,' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume False
