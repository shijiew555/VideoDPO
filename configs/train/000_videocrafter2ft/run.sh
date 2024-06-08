export TOKENIZERS_PARALLELISM=false

current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="dpo-10k"                            # experiment name 
CONFIG='configs/train/000_videocrafter2ft/config.yaml' # experiment config 
LOGDIR="./results/dpo-10k-0530"                         # experiment saving directory all should under subfolder so that won't be copied to codeversion

# run
python scripts/train.py \
-t --devices '4' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume False


# python -m torch.distributed.run \
# --nnodes=1 \
# --nproc_per_node=1 \
# scripts/train.py \
# -t --devices '0,1,2,3' \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$current_time"_$EXPNAME \
# --logdir $LOGDIR \
# --auto_resume True